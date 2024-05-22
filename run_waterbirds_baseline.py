import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
import skimage, torch, torchvision
import pickle 
import cv2
from tqdm import tqdm
import random
from torch.utils.data import Dataset
import os
import argparse 
from torcheval.metrics import BinaryAUROC
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import v2
from Waterbirds_copy import Waterbirds


parser = argparse.ArgumentParser(description='Waterbirds baseline training.')
parser.add_argument('--output_dir', type=str, default="")
parser.add_argument('--use_cutmix', action="store_true")
parser.add_argument('--use_mixup', action="store_true")
parser.add_argument('--alpha', type=float, default=1., help='hyperparameter for cutmix and mixup augmentation')
parser.add_argument('--train_model', action="store_true")
parser.add_argument('--transfer_model', action="store_true")
parser.add_argument('--eval_model', action="store_true")
parser.add_argument('--freeze_encoder', action="store_true")
parser.add_argument('--use_diffusion_images', action="store_true")
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--checkpoint_epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--ablation_num', type=int, default=5)
parser.add_argument('--gen_ablation_num', type=int, default=None)
args = parser.parse_args()

transform = torchvision.transforms.Compose([v2.CenterCrop(224),
                                            v2.RandomHorizontalFlip(),
                                            v2.ToImage(),
                                            v2.ToDtype(torch.float32, scale=True),
                                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                                            ])

with open("waterbirds_preprocessed_datasets_classifier.pkl", "rb") as f: 
    preprocessed_datasets = pickle.load(f)

train_dataloader = torch.utils.data.DataLoader(preprocessed_datasets['train'], batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(preprocessed_datasets['val'], batch_size=8, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(preprocessed_datasets['test'], batch_size=8, shuffle=False)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

output_dir = args.output_dir
if not os.path.exists(output_dir): 
    os.makedirs(output_dir)

if "cutmix" in output_dir: 
    args.use_cutmix = True 
if "mixup" in output_dir: 
    args.use_mixup = True

device = "cuda"

if args.train_model: 
    if args.use_diffusion_images:
        gen_dir = "/mnt/scratch-lids/scratch/qixuanj/waterbird_generated_images/waterbirds_finetune_sd_token2_816_dreambooth/target100_background/strength1.0"
        gen_ds = Waterbirds(gen_dir, "train", transform=transform)

        # Previous set of results is with mix strength
        # gen_dir = "/mnt/scratch-lids/scratch/qixuanj/waterbird_generated_images/waterbirds_finetune_sd_token2_816_dreambooth/mix_strength"
        # gen_ds = Waterbirds(gen_dir, "gen", transform=transform)
        
        # gen_ds = torch.utils.data.Subset(gen_ds, random.sample(list(range(len(gen_ds))), 100))
                                         
        # gen_dir2 = "/mnt/scratch-lids/scratch/qixuanj/waterbird_generated_images/waterbirds_finetune_sd_token2_816/strength0.7"
        # gen_ds2 = Waterbirds(gen_dir2, "gen", transform=transform)
        # gen_ds2 = torch.utils.data.Subset(gen_ds2, random.sample(list(range(len(gen_ds2))), 100)) 
                                          
        mix_dataset = torch.utils.data.ConcatDataset([preprocessed_datasets['train'], 
                                                      gen_ds, 
                                                      # gen_ds2,
                                                     ])
        train_dataloader = torch.utils.data.DataLoader(mix_dataset, batch_size=64, shuffle=True)
    
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc =  torch.nn.Linear(2048, 1)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.BCELoss()

    if args.use_cutmix or args.use_mixup:
        print("Use cutmix or mixup!")
        cutmix = v2.CutMix(num_classes=2, alpha=args.alpha)
        mixup = v2.MixUp(num_classes=2, alpha=args.alpha)
    
    train_aucs = []
    val_aucs = [] 
    
    if args.freeze_encoder: 
        for param in model.parameters():
            param.requires_grad = False
        # Only train classifier 
        for param in model.fc.parameters():
            param.requires_grad = True
                
    avg_loss = []
    for epoch in range(args.epochs):
        model.train()
        task_outputs=[]
        task_targets=[]
        
        t = tqdm(train_dataloader)
        for batch_idx, sample in enumerate(t): 
            optimizer.zero_grad()
            images, targets = sample[0], sample[1]
            if args.use_cutmix: 
                images, converted_targets = cutmix(images, targets)
                converted_targets = converted_targets[:, 1]
                converted_targets = converted_targets.to(device)
            if args.use_mixup: 
                images, converted_targets = mixup(images, targets)
                converted_targets = converted_targets[:, 1]
                converted_targets = converted_targets.to(device)
            
            images = images.to("cuda")
            targets = targets.to("cuda")
            outputs = model(images)
            sigmoid = torch.nn.Sigmoid()
            outputs = sigmoid(outputs)
            outputs = outputs.squeeze(1)
            
            targets = targets.to(torch.float32)
            if args.use_cutmix or args.use_mixup: 
                converted_targets = converted_targets.to(torch.float32)
                loss = criterion(outputs, converted_targets)
            else: 
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.detach().cpu().numpy())
            task_outputs.append(outputs.detach().cpu().numpy())
            task_targets.append(targets.detach().cpu().numpy())
            t.set_description(f'Epoch {epoch + 1} - Loss = {np.mean(avg_loss):4.4f}')
            
        task_outputs = np.concatenate(task_outputs)
        task_targets = np.concatenate(task_targets)
        auc = BinaryAUROC()
        auc.update(torch.Tensor(task_outputs), torch.Tensor(task_targets))
        auc_result = auc.compute()
        train_aucs.append(auc_result)
        
        if (epoch + 1) % args.checkpoint_epochs == 0: 
            print("checkpoint!")
            model.eval()
            t = tqdm(val_dataloader)
            val_task_outputs=[]
            val_task_targets=[]
            with torch.no_grad():
                for batch_idx, sample in enumerate(t):
                    images, targets = sample[0], sample[1]
                    images = images.to("cuda")
                    targets = targets.to("cuda")
                    outputs = model(images)
                    sigmoid = torch.nn.Sigmoid()
                    outputs = sigmoid(outputs)
                    outputs = outputs.squeeze(1)
                    targets = targets.to(torch.float32)
                    val_task_outputs.append(outputs.detach().cpu().numpy())
                    val_task_targets.append(targets.detach().cpu().numpy())
            val_task_outputs = np.concatenate(val_task_outputs)
            val_task_targets = np.concatenate(val_task_targets)
            auc = BinaryAUROC()
            auc.update(torch.Tensor(val_task_outputs), torch.Tensor(val_task_targets))
            val_auc_result = auc.compute()
            val_aucs.append(val_auc_result)
            torch.save(model, f"{output_dir}/checkpoint{epoch + 1}.pt")
            print(f'Val Epoch {epoch + 1} - Avg AUC = {val_auc_result:4.4f}')
        print(f'Train Epoch {epoch + 1} - Avg AUC = {auc_result:4.4f}')
        
    torch.save(model, f"{output_dir}/model.pt")
    with open(f"{output_dir}/train_aucs.pkl", "wb") as f: 
        pickle.dump(train_aucs, f) 
    with open(f"{output_dir}/val_aucs.pkl", "wb") as f: 
        pickle.dump(val_aucs, f) 

if args.transfer_model: 
    if args.use_diffusion_images:
        gen_dir = "/mnt/scratch-lids/scratch/qixuanj/waterbird_generated_images/waterbirds_finetune_sd_token2_816_dreambooth/target100_background/strength1.0"
        gen_ds = Waterbirds(gen_dir, "train", transform=transform)
        if args.gen_ablation_num: 
            gen_ds = torch.utils.data.Subset(gen_ds, gen_ds.df.sample(n=args.gen_ablation_num, random_state=0).index)

        # gen_dir = "/mnt/scratch-lids/scratch/qixuanj/waterbird_generated_images/waterbirds_finetune_sd_token2_816_dreambooth/source"
        # gen_ds = Waterbirds(gen_dir, "train", transform=transform)
        
        # gen_dir = "/mnt/scratch-lids/scratch/qixuanj/waterbird_generated_images/waterbirds_finetune_sd_token2_816_dreambooth/mix_strength"
        # gen_ds = Waterbirds(gen_dir, "gen", transform=transform)
        # gen_ds = torch.utils.data.Subset(gen_ds, random.sample(list(range(len(gen_ds))), 100))
                                         
        # gen_dir2 = "/mnt/scratch-lids/scratch/qixuanj/waterbird_generated_images/waterbirds_finetune_sd_token2_816/strength0.7"
        # gen_ds2 = Waterbirds(gen_dir2, "gen", transform=transform)
        # gen_ds2 = torch.utils.data.Subset(gen_ds2, random.sample(list(range(len(gen_ds2))), 100))
                                          
        mix_dataset = torch.utils.data.ConcatDataset([preprocessed_datasets['train'], 
                                                      # preprocessed_datasets['extra'],
                                                      # preprocessed_datasets['extra_group1'],
                                                      # preprocessed_datasets['extra_group2'],
                                                      gen_ds, 
                                                      # gen_ds2,
                                                     ])
        mix_dataloader = torch.utils.data.DataLoader(mix_dataset, batch_size=64, shuffle=True)
    else:
        print(f"Ablation {args.ablation_num}")
        mix_dataset = torch.utils.data.ConcatDataset([
                                                     preprocessed_datasets['train'], 
                                                    preprocessed_datasets[f'extra_{args.ablation_num}_group1'],
                                                    preprocessed_datasets[f'extra_{args.ablation_num}_group2'],
                                                     ])
        mix_dataloader = torch.utils.data.DataLoader(mix_dataset, batch_size=64, shuffle=True)

    savedir = output_dir + "/transfer"
    if not os.path.exists(savedir): 
        os.makedirs(savedir)
        
    # with open(f"{output_dir}/val_aucs.pkl", "rb") as f: 
    #     val_aucs = pickle.load(f)
    # val_aucs = [x.item() for x in val_aucs] 
    
    # epochs_map = np.arange(10, 101, 10)
    # checkpoint = str(epochs_map[np.argmax(np.array(val_aucs))])
    # model = torch.load(f"{output_dir}/checkpoint{checkpoint}.pt")

    # New Transfer
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc =  torch.nn.Linear(2048, 1)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.BCELoss()

    if args.use_cutmix or args.use_mixup:
        cutmix = v2.CutMix(num_classes=2, alpha=args.alpha)
        mixup = v2.MixUp(num_classes=2, alpha=args.alpha)
    
    train_aucs = []
    val_aucs = [] 
    
    model.to("cuda")
    
    if args.freeze_encoder: 
        for param in model.parameters():
            param.requires_grad = False
        # Only train classifier 
        for param in model.fc.parameters():
            param.requires_grad = True
                
    avg_loss = []
    for epoch in range(args.epochs):
        model.train()
        task_outputs=[]
        task_targets=[]
        
        t = tqdm(mix_dataloader)
        for batch_idx, sample in enumerate(t): 
            optimizer.zero_grad()
            images, targets = sample[0], sample[1]
            if args.use_cutmix: 
                images, converted_targets = cutmix(images, targets)
                converted_targets = converted_targets[:, 1]
                converted_targets = converted_targets.to(device)
            if args.use_mixup: 
                images, converted_targets = mixup(images, targets)
                converted_targets = converted_targets[:, 1]
                converted_targets = converted_targets.to(device)
            
            images = images.to("cuda")
            targets = targets.to("cuda")
            outputs = model(images)
            sigmoid = torch.nn.Sigmoid()
            outputs = sigmoid(outputs)
            outputs = outputs.squeeze(1)
            
            targets = targets.to(torch.float32)
            if args.use_cutmix or args.use_mixup: 
                converted_targets = converted_targets.to(torch.float32)
                loss = criterion(outputs, converted_targets)
            else: 
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.detach().cpu().numpy())
            task_outputs.append(outputs.detach().cpu().numpy())
            task_targets.append(targets.detach().cpu().numpy())
            t.set_description(f'Extra Epoch {epoch + 1} - Loss = {np.mean(avg_loss):4.4f}')
            
        task_outputs = np.concatenate(task_outputs)
        task_targets = np.concatenate(task_targets)
        auc = BinaryAUROC()
        auc.update(torch.Tensor(task_outputs), torch.Tensor(task_targets))
        auc_result = auc.compute()
        train_aucs.append(auc_result)
        
        if (epoch + 1) % args.checkpoint_epochs == 0: 
            print("checkpoint!")
            model.eval()
            t = tqdm(val_dataloader)
            val_task_outputs=[]
            val_task_targets=[]
            with torch.no_grad():
                for batch_idx, sample in enumerate(t):
                    images, targets = sample[0], sample[1]
                    images = images.to("cuda")
                    targets = targets.to("cuda")
                    outputs = model(images)
                    sigmoid = torch.nn.Sigmoid()
                    outputs = sigmoid(outputs)
                    outputs = outputs.squeeze(1)
                    targets = targets.to(torch.float32)
                    val_task_outputs.append(outputs.detach().cpu().numpy())
                    val_task_targets.append(targets.detach().cpu().numpy())
            val_task_outputs = np.concatenate(val_task_outputs)
            val_task_targets = np.concatenate(val_task_targets)
            auc = BinaryAUROC()
            auc.update(torch.Tensor(val_task_outputs), torch.Tensor(val_task_targets))
            val_auc_result = auc.compute()
            val_aucs.append(val_auc_result)
            torch.save(model, f"{savedir}/checkpoint{epoch + 1}.pt")
            print(f'Val Epoch {epoch + 1} - Avg AUC = {val_auc_result:4.4f}')
        print(f'Epoch {epoch + 1} - Avg AUC = {auc_result:4.4f}')
        
    torch.save(model, f"{savedir}/model.pt")
    with open(f"{savedir}/train_aucs.pkl", "wb") as f: 
        pickle.dump(train_aucs, f) 
    with open(f"{savedir}/val_aucs.pkl", "wb") as f: 
        pickle.dump(val_aucs, f) 

if args.eval_model: 
    epochs = np.arange(args.checkpoint_epochs, args.epochs + 1, args.checkpoint_epochs)

    with open(f"{output_dir}/val_aucs.pkl", "rb") as f: 
        val_aucs = pickle.load(f)
    val_aucs = [x.item() for x in val_aucs] 
    
    # Get best checkpoint by highest val AUC 
    epochs_map = np.arange(args.checkpoint_epochs, args.epochs + 1, args.checkpoint_epochs)
    checkpoint = str(epochs_map[np.argmax(np.array(val_aucs))])
    print(f"Best checkpoint is {checkpoint} with val AUC of {np.max(np.array(val_aucs))}")
    
    model = torch.load(f"{output_dir}/checkpoint{checkpoint}.pt")
    model.eval()
    t = tqdm(test_dataloader)
    
    test_task_outputs=[]
    test_task_targets=[]
    with torch.no_grad():
        for batch_idx, sample in enumerate(t):
            images, targets = sample[0], sample[1]
            images = images.to("cuda")
            targets = targets.to("cuda")
            outputs = model(images)
            sigmoid = torch.nn.Sigmoid()
            outputs = sigmoid(outputs)
            outputs = outputs.squeeze(1)
            targets = targets.to(torch.float32)
            test_task_outputs.append(outputs.detach().cpu().numpy())
            test_task_targets.append(targets.detach().cpu().numpy())
    test_task_outputs = np.concatenate(test_task_outputs)
    test_task_targets = np.concatenate(test_task_targets)
    
    auc = BinaryAUROC()
    auc.update(torch.Tensor(test_task_outputs), torch.Tensor(test_task_targets))
    test_auc_result = auc.compute()
    print("Test AUC of {}".format(test_auc_result))
    
    with open(f"{output_dir}/test_auc.pkl", "wb") as f: 
        pickle.dump(test_auc_result, f) 
    with open(f"{output_dir}/test_outputs.pkl", "wb") as f: 
        pickle.dump(test_task_outputs, f) 
    with open(f"{output_dir}/test_targets.pkl", "wb") as f: 
        pickle.dump(test_task_targets, f) 