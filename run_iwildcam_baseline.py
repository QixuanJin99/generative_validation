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
from torcheval.metrics import MulticlassAUROC
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import v2
from iwildcam_dataset import iWildCamDataset
import random


parser = argparse.ArgumentParser(description='iWildCam baseline training.')
parser.add_argument('--shift', type=str, required=True)
parser.add_argument('--output_dir', type=str, default="")
parser.add_argument('--use_cutmix', action="store_true")
parser.add_argument('--use_mixup', action="store_true")
parser.add_argument('--alpha', type=float, default=1., help='hyperparameter for cutmix and mixup augmentation')
parser.add_argument('--train_model', action="store_true")
parser.add_argument('--transfer_model', action="store_true")
parser.add_argument('--eval_model', action="store_true")
parser.add_argument('--freeze_encoder', action="store_true")
parser.add_argument('--use_diffusion_images', action="store_true")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--checkpoint_epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model_type', type=str, default='resnet50')
parser.add_argument('--gen_strength', type=str, default='gen_0.7')
parser.add_argument('--version', type=str, default='copy2')
parser.add_argument('--gen_version', type=str, default='copy2')
parser.add_argument('--filter', type=str, default=None)
args = parser.parse_args()

# Important to not crop anything as the spurious signal is often on the edges
transform = torchvision.transforms.Compose([# v2.CenterCrop(224),
                                            v2.Resize((224, 224)), 
                                            v2.RandomHorizontalFlip(),
                                            v2.ToImage(),
                                            v2.ToDtype(torch.float32, scale=True),
                                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                                            ])

filepath = "/mnt/scratch-lids/scratch/qixuanj/iwildcam_subset_organized"
train_dataset = iWildCamDataset(file_path=filepath, 
                                shift = args.shift, 
                                split = "train", 
                                transform=transform, 
                               version=args.version)
extra_dataset = iWildCamDataset(file_path=filepath, 
                                shift = args.shift, 
                                split = "extra", 
                                transform=transform, 
                               version=args.version)
print("Extra size: {}".format(len(extra_dataset)))
val_dataset = iWildCamDataset(file_path=filepath, 
                                shift = args.shift, 
                                split = "val", 
                                transform=transform, 
                             version=args.version)
test_dataset = iWildCamDataset(file_path=filepath, 
                                shift = args.shift, 
                                split = "test", 
                                transform=transform, 
                              version=args.version)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

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
        if args.gen_strength == "mix_branches": 
            gen_ds1 = iWildCamDataset("/mnt/scratch-lids/scratch/qixuanj/iwildcam_subset_organized", 
                                shift=args.shift, 
                                split="branches_strength0.9_filter2", 
                                transform=transform, 
                                version="copy2",)
            gen_ds2 = iWildCamDataset("/mnt/scratch-lids/scratch/qixuanj/iwildcam_subset_organized", 
                                shift=args.shift, 
                                split="no_branches_strength0.9_0_filter2", 
                                transform=transform, 
                                version="copy2",)
            indices = random.choices(range(len(gen_ds1)), k=50)
            
            gen_ds = torch.utils.data.ConcatDataset([torch.utils.data.Subset(gen_ds1, indices), 
                                                     gen_ds2,])
        else:   
            gen_ds = iWildCamDataset("/mnt/scratch-lids/scratch/qixuanj/iwildcam_subset_organized", 
                                    shift=args.shift, 
                                    split=args.gen_strength, 
                                    transform=transform, 
                                    version=args.gen_version, 
                                    filter=args.filter)
        mix_dataset = torch.utils.data.ConcatDataset([train_dataset, 
                                                      gen_ds, 
                                                     ])
        train_dataloader = torch.utils.data.DataLoader(mix_dataset, batch_size=8, shuffle=True)
    if args.model_type == 'resnet50': 
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc =  torch.nn.Linear(2048, 7)
    elif args.model_type == 'densenet121':
        model = torchvision.models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
        model.classifier = torch.nn.Linear(1024, 7)
    else: 
        raise Exception("Invalid model type")
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    if args.use_cutmix or args.use_mixup:
        print("Use cutmix or mixup!")
        cutmix = v2.CutMix(num_classes=7, alpha=args.alpha)
        mixup = v2.MixUp(num_classes=7, alpha=args.alpha)
    
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
                converted_targets = converted_targets.to(torch.float32).to(device)
            if args.use_mixup: 
                images, converted_targets = mixup(images, targets)
                converted_targets = converted_targets.to(torch.float32).to(device)
            
            images = images.to("cuda")
            outputs = model(images)
            
            # targets = targets.to(torch.float32)
            targets = targets.type(torch.LongTensor)
            targets = targets.to("cuda")
            
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
        auc = MulticlassAUROC(num_classes=7)
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
                    outputs = model(images)

                    # targets = targets.to(torch.float32)
                    targets = targets.type(torch.LongTensor)
                    targets = targets.to("cuda")
                    
                    val_task_outputs.append(outputs.detach().cpu().numpy())
                    val_task_targets.append(targets.detach().cpu().numpy())
            val_task_outputs = np.concatenate(val_task_outputs)
            val_task_targets = np.concatenate(val_task_targets)
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
        if args.gen_strength == "gen_combine": 
            gen_ds1 = iWildCamDataset("/mnt/scratch-lids/scratch/qixuanj/iwildcam_subset_organized", 
                                shift=args.shift, 
                                split="gen_0.7", 
                                transform=transform, 
                                version=args.gen_version, 
                                filter=args.filter)
            gen_ds2 = iWildCamDataset("/mnt/scratch-lids/scratch/qixuanj/iwildcam_subset_organized", 
                                shift=args.shift, 
                                split="gen_0.9", 
                                transform=transform, 
                                version=args.gen_version, 
                                filter=args.filter)
            gen_ds = torch.utils.data.ConcatDataset([gen_ds1, gen_ds2])
        else:
            gen_ds = iWildCamDataset("/mnt/scratch-lids/scratch/qixuanj/iwildcam_subset_organized", 
                                    shift=args.shift, 
                                    split=args.gen_strength, 
                                    transform=transform, 
                                    version=args.gen_version, 
                                    filter=args.filter)
        
        mix_dataset = torch.utils.data.ConcatDataset([train_dataset, 
                                                      extra_dataset,
                                                      gen_ds, 
                                                     ])
        mix_dataloader = torch.utils.data.DataLoader(mix_dataset, batch_size=8, shuffle=True)
    else:
        mix_dataset = torch.utils.data.ConcatDataset([train_dataset, 
                                                      extra_dataset,])
        mix_dataloader = torch.utils.data.DataLoader(mix_dataset, batch_size=8, shuffle=True)

    savedir = output_dir + "/transfer"
    if not os.path.exists(savedir): 
        os.makedirs(savedir)
        
    with open(f"{output_dir}/val_aucs.pkl", "rb") as f: 
        val_aucs = pickle.load(f)
    val_aucs = [x.item() for x in val_aucs] 
    
    epochs_map = np.arange(10, 101, 10)
    checkpoint = str(epochs_map[np.argmax(np.array(val_aucs))])
    model = torch.load(f"{output_dir}/checkpoint{checkpoint}.pt")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    if args.use_cutmix or args.use_mixup:
        cutmix = v2.CutMix(num_classes=7, alpha=args.alpha)
        mixup = v2.MixUp(num_classes=7, alpha=args.alpha)
    
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
                converted_targets = converted_targets.to(torch.float32).to(device)
            if args.use_mixup: 
                images, converted_targets = mixup(images, targets)
                converted_targets = converted_targets.to(torch.float32).to(device)
            
            images = images.to("cuda")
            outputs = model(images)
            
            # targets = targets.to(torch.float32)
            targets = targets.type(torch.LongTensor)
            targets = targets.to("cuda")
            
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
        auc = MulticlassAUROC(num_classes=7)
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
                    outputs = model(images)
                    
                    targets = targets.type(torch.LongTensor)
                    targets = targets.to("cuda")
                    
                    val_task_outputs.append(outputs.detach().cpu().numpy())
                    val_task_targets.append(targets.detach().cpu().numpy())
            val_task_outputs = np.concatenate(val_task_outputs)
            val_task_targets = np.concatenate(val_task_targets)
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
            
            targets = targets.type(torch.LongTensor)
            test_task_outputs.append(outputs.detach().cpu().numpy())
            test_task_targets.append(targets.detach().cpu().numpy())
    test_task_outputs = np.concatenate(test_task_outputs)
    test_task_targets = np.concatenate(test_task_targets)
    
    auc = MulticlassAUROC(num_classes=7)
    auc.update(torch.Tensor(test_task_outputs), torch.Tensor(test_task_targets))
    test_auc_result = auc.compute()
    print("Test AUC of {}".format(test_auc_result))
    
    with open(f"{output_dir}/test_auc.pkl", "wb") as f: 
        pickle.dump(test_auc_result, f) 
    with open(f"{output_dir}/test_outputs.pkl", "wb") as f: 
        pickle.dump(test_task_outputs, f) 
    with open(f"{output_dir}/test_targets.pkl", "wb") as f: 
        pickle.dump(test_task_targets, f) 