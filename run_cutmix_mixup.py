from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from PIL import Image
import pickle 
import cv2
import torchxrayvision as xrv
import skimage, torch, torchvision
import pickle
import argparse
import os
from torchvision.transforms import v2
import random

def get_devices(gpus):
    if len(gpus) == 0:
        device_ids = None
        device = torch.device('cpu')
        print('Warning! Computing on CPU')
    elif len(gpus) == 1:
        device_ids = None
        device = torch.device('cuda:' + str(gpus[0]))
    else:
        device_ids = [int(i) for i in gpus]
        device = torch.device('cuda:' + str(min(device_ids)))
    return device, device_ids


def convert_output(model_name, outputs): 
    # Function that convert the model output to standard 8 labels 
    output_mapping = {
        "mimic": [0, 10, 1, 4, 14, 8, 3], 
        "chexpert": [0, 10, 1, 4, 14, 8, 3], 
        "padchest": [0, 10, 1, 4, 8, 3], 
        "nih": [0, 10, 1, 4, 8, 3], 
        "all": [0, 10, 1, 4, 14, 8, 3],
    }
    pathologies = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
               "Lesion", "Pneumonia", "Pneumothorax", "No Finding"]
    new_outputs = outputs[:, output_mapping[model_name]]

    if model_name == "padchest" or model_name == "nih": 
        lesion = torch.max(outputs[:, 11], outputs[:, 12])
        new_outputs = torch.hstack((new_outputs[:, :4], lesion.unsqueeze(1), new_outputs[:, 4:]))

    # Append "No Findings" output as last column 
    no_finding_output = outputs[:, [i for i, x in enumerate(pathologies) if x != '']]
    # Max probability over all class with positive finding, then take the inverse 
    no_finding = 1. - no_finding_output.max(axis=1)[0]
    new_outputs = torch.hstack((new_outputs, no_finding.unsqueeze(1)))
    return new_outputs

# Specific to CXR labelled in order of pathologies 
def convert_target_cxr(labels): 
    pos_idx = np.argwhere(labels == 1)
    pos_idx = pd.DataFrame(pos_idx)
    assert(pos_idx.shape[1] == 2)
    pos_idx = pos_idx.groupby(0).apply(lambda x: x.sample(1, random_state=0)).reset_index(drop=True)
    converted_labels = pd.DataFrame(index=range(labels.shape[0]), columns = ["label"])
    converted_labels.loc[pos_idx[0], "label"] = pos_idx[1].values
    # Fill in as No Finding 
    converted_labels = torch.from_numpy(converted_labels.fillna(7).values.squeeze()).to(torch.int64)
    return converted_labels

def valid_test_epoch(name, epoch, model, model_name, device, data_loader, 
                     criterion, num_classes=8, limit=None, 
                     resolution=512):
    model.to(device)
    model.eval()

    avg_loss = []
    task_outputs={}
    task_targets={}
    for task in range(data_loader.dataset[0]["lab"].shape[0]):
        task_outputs[task] = []
        task_targets[task] = []
    
    with torch.no_grad():
        t = tqdm(data_loader)
        for batch_idx, samples in enumerate(t):
            if limit and (batch_idx > limit):
                print("breaking out")
                break
            
            indices = samples["idx"]
            images = samples["img"]
            targets = samples["lab"]

            # Convert to three channels 
            images = torch.cat((images, images, images), 1)
        
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            # outputs = convert_output(model_name, outputs)
            
            loss = torch.zeros(1).to(device).double()
            for task in range(targets.shape[1]):
                task_output = outputs[:,task]
                task_target = targets[:,task]

                mask = ~torch.isnan(task_target)
                task_output = task_output[mask]
                task_target = task_target[mask]
                if len(task_target) > 0:
                    loss += criterion(task_output.double(), task_target.double())
                
                task_outputs[task].append(task_output.detach().cpu().numpy())
                task_targets[task].append(task_target.detach().cpu().numpy())

            loss = loss.sum()
            
            avg_loss.append(loss.detach().cpu().numpy())
            t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')
            
        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])
    
        task_aucs = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task]))> 1:
                task_auc = roc_auc_score(task_targets[task], task_outputs[task])
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')
    return auc, task_aucs, task_outputs, task_targets


def train_epoch(name, epoch, model, model_name, device, data_loader, 
                     criterion, optimizer, num_classes=8,
                     use_cutmix=False, use_mixup=False, 
                     resolution=512, freeze_encoder=True, limit=None, alpha=1.):
    model.to(device)
    model.train()
    
    if freeze_encoder: 
        for param in model.parameters():
            param.requires_grad = False
        # Only train classifier 
        for param in model.classifier.parameters():
            param.requires_grad = True
    else: 
        for param in model.parameters():
            param.requires_grad = True
    
    if use_cutmix: 
        cutmix = v2.CutMix(num_classes=num_classes, alpha=alpha)
    if use_mixup: 
        mixup = v2.MixUp(num_classes=num_classes, alpha=alpha) 

    avg_loss = []
    task_outputs={}
    task_targets={}
    for task in range(data_loader.dataset[0]["lab"].shape[0]):
        task_outputs[task] = []
        task_targets[task] = []
         
    t = tqdm(data_loader)
    for batch_idx, samples in enumerate(t):
        if limit and (batch_idx > limit):
                print("breaking out")
                break
                
        optimizer.zero_grad()
        indices = samples["idx"]
        images = samples["img"]
        targets = samples["lab"]

        if use_cutmix: 
            converted_targets = convert_target_cxr(targets.numpy())
            images, converted_targets = cutmix(images, converted_targets)
            converted_targets = converted_targets.to(device)
        if use_mixup: 
            converted_targets = convert_target_cxr(targets.numpy())
            images, converted_targets = mixup(images, converted_targets) 
            converted_targets = converted_targets.to(device)
            
        # Convert to three channels 
        images = torch.cat((images, images, images), 1)

        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        # outputs = convert_output(model_name, outputs)

        loss = torch.zeros(1).to(device).double()
        for task in range(targets.shape[1]):
            task_output = outputs[:,task]
            if use_cutmix or use_mixup: 
                converted_task_target = converted_targets[:,task]
                if len(converted_task_target) > 0:
                    loss += criterion(task_output.double(), converted_task_target.double())
                
                task_target = targets[:,task]
                mask = ~torch.isnan(task_target)
                task_output = task_output[mask]
                task_target = task_target[mask]
            else: 
                task_target = targets[:,task]
                mask = ~torch.isnan(task_target)
                task_output = task_output[mask]
                task_target = task_target[mask]
                if len(task_target) > 0:
                    loss += criterion(task_output.double(), task_target.double())

            task_outputs[task].append(task_output.detach().cpu().numpy())
            task_targets[task].append(task_target.detach().cpu().numpy())

        loss = loss.sum()
        
        # Backprop
        loss.backward()
        optimizer.step()

        avg_loss.append(loss.detach().cpu().numpy())
        t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')

    for task in range(len(task_targets)):
        task_outputs[task] = np.concatenate(task_outputs[task])
        task_targets[task] = np.concatenate(task_targets[task])

    task_aucs = []
    for task in range(len(task_targets)):
        if len(np.unique(task_targets[task]))> 1:
            task_auc = roc_auc_score(task_targets[task], task_outputs[task])
            task_aucs.append(task_auc)
        else:
            task_aucs.append(np.nan)

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')
    return model, auc, task_aucs



if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Train or eval baseline cutmix and mixup.')
    parser.add_argument('--model_name', type=str, default="mimic")
    parser.add_argument('--dataset_name', type=str, default="mimic")
    parser.add_argument('--train_num', type=int, default=100)
    parser.add_argument('--class_balanced', action="store_true")
    parser.add_argument('--output_dir', type=str, default="")
    parser.add_argument('--use_cutmix', action="store_true")
    parser.add_argument('--use_mixup', action="store_true")
    parser.add_argument('--train_model', action="store_true")
    parser.add_argument('--transfer_model', action="store_true")
    parser.add_argument('--eval_train_model', action="store_true")
    parser.add_argument('--eval_model', action="store_true")
    parser.add_argument('--freeze_encoder', action="store_true")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--checkpoint_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = args.output_dir
    model_name = args.model_name
    dataset_name = args.dataset_name
    
    device, device_ids = get_devices([0])

    if args.debug: 
        limit = 10
    else: 
        limit = None
    
    criterion = torch.nn.BCEWithLogitsLoss()
    if args.train_model:
        model = torchvision.models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
        model.classifier = torch.nn.Linear(1024, 8)
        
        with open("split_datasets_balanced_v2.pkl", "rb") as f: 
            split_datasets = pickle.load(f)
        base_dataset = split_datasets[model_name]["train"]
        val_dataset = split_datasets[model_name]["val"]
            
        base_train_dataloader = torch.utils.data.DataLoader(base_dataset, batch_size=16, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        print("Training model")
        train_log = {'auc': [], 'task_aucs': []}
        val_log = {'auc': [], 'task_aucs': []}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        if not os.path.exists(f"{output_dir}/train/models/{args.seed}"): 
            os.makedirs(f"{output_dir}/train/models/{args.seed}") 
        if not os.path.exists(f"{output_dir}/train/logs/{args.seed}"): 
            os.makedirs(f"{output_dir}/train/logs/{args.seed}")
        
        for e in range(args.epochs): 
            model, auc, task_aucs = train_epoch("train", e, model, model_name, 
                                                              "cuda", base_train_dataloader, 
                                                              criterion, optimizer, 
                                                              num_classes=8,
                                                              use_mixup=args.use_mixup, 
                                                              use_cutmix=args.use_cutmix, 
                                                              freeze_encoder=args.freeze_encoder,
                                                              alpha=args.alpha, 
                                                               limit=limit)
            if (e + 1) % args.checkpoint_epochs == 0: 
                val_auc, val_task_aucs, _, _ = valid_test_epoch("val", 0, model, model_name, 
                                                                  "cuda", val_dataloader, 
                                                                  torch.nn.BCEWithLogitsLoss(), 
                                                                  num_classes=8,
                                                               limit=limit)
                val_log['auc'].append(val_auc)
                val_log['task_aucs'].append(val_task_aucs)
                torch.save(model, f"{output_dir}/train/models/{args.seed}/{model_name}_epoch{e + 1}_model.pt")
            train_log['auc'].append(auc)
            train_log['task_aucs'].append(task_aucs)
            
        # Save model
        torch.save(model, f"{output_dir}/train/models/{args.seed}/{model_name}_epoch{args.epochs}_model.pt")
        # Save training log 
        with open(f"{output_dir}/train/logs/{args.seed}/{model_name}_epoch{args.epochs}_log.pkl", "wb") as f: 
            pickle.dump(train_log, f)
        with open(f"{output_dir}/train/logs/{args.seed}/{model_name}_epoch{args.epochs}_val_log.pkl", "wb") as f: 
            pickle.dump(val_log, f)

    if args.transfer_model: 
        with open("split_datasets_balanced_v2.pkl", "rb") as f: 
            split_datasets = pickle.load(f)
        with open(f"transfer_dataset_seed{args.seed}.pkl", "rb") as f: 
            transfer_dataset = pickle.load(f)

        with open(f"{args.output_dir}/train/logs/{args.seed}/{model_name}_epoch10_val_log.pkl", "rb") as f: 
            val_log = pickle.load(f)
            
        # Select model by best validation accuracy 
        val_aucs = [x.item() for x in val_log['auc']]
        epochs_map = np.arange(2, 11, 2)
        checkpoint = str(epochs_map[np.argmax(np.array(val_aucs))])

        model_path = f"{args.output_dir}/train/models/{args.seed}/{model_name}_epoch{checkpoint}_model.pt"
        model = torch.load(model_path)

        source_dataset = split_datasets[model_name]["train"]
        val_dataset = split_datasets[dataset_name]["val"]
        
        if args.class_balanced:
            keyword = "balanced"
            transfer_dataset = transfer_dataset[dataset_name][args.train_num]['balanced_ds']
        else: 
            keyword = "match"
            transfer_dataset = transfer_dataset[dataset_name][args.train_num]['match_ds']
        mix_dataset = torch.utils.data.ConcatDataset([source_dataset, transfer_dataset])
            
        mix_dataloader = torch.utils.data.DataLoader(mix_dataset, batch_size=8, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        if not os.path.exists(f"{output_dir}/transfer/models/{args.train_num}/{args.seed}"): 
            os.makedirs(f"{output_dir}/transfer/models/{args.train_num}/{args.seed}") 
        if not os.path.exists(f"{output_dir}/transfer/logs/{args.train_num}/{args.seed}"): 
            os.makedirs(f"{output_dir}/transfer/logs/{args.train_num}/{args.seed}") 
        
        print("Transfer model")
        train_log = {'auc': [], 'task_aucs': []}
        val_log = {'auc': [], 'task_aucs': []}
        for e in range(args.epochs): 
            model, auc, task_aucs = train_epoch("train", e+1, model, model_name, 
                                                          "cuda", mix_dataloader, 
                                                          criterion, optimizer, 
                                                          num_classes=8,
                                                          use_mixup=args.use_mixup, 
                                                          use_cutmix=args.use_cutmix, 
                                                          freeze_encoder=args.freeze_encoder, 
                                                           alpha=args.alpha, 
                                                           limit=limit)
            if (e + 1) % args.checkpoint_epochs == 0: 
                val_auc, val_task_aucs, _, _ = valid_test_epoch("val", 0, model, model_name, 
                                                                  "cuda", val_dataloader, 
                                                                  torch.nn.BCEWithLogitsLoss(), 
                                                                  num_classes=8,
                                                                  limit=limit)
                val_log['auc'].append(val_auc)
                val_log['task_aucs'].append(val_task_aucs)
                torch.save(model, f"{output_dir}/transfer/models/{args.train_num}/{args.seed}/{model_name}_{dataset_name}_{keyword}_epoch{e + 1}_model.pt")
                
            train_log['auc'].append(auc)
            train_log['task_aucs'].append(task_aucs)
            
        # Save model
        torch.save(model, f"{output_dir}/transfer/models/{args.train_num}/{args.seed}/{model_name}_{dataset_name}_{keyword}_epoch{args.epochs}_model.pt")
        # Save training log 
        with open(f"{output_dir}/transfer/logs/{args.train_num}/{args.seed}/{model_name}_{dataset_name}_{keyword}_epoch{args.epochs}_log.pkl", "wb") as f: 
            pickle.dump(train_log, f)
        with open(f"{output_dir}/transfer/logs/{args.train_num}/{args.seed}/{model_name}_{dataset_name}_{keyword}_epoch{args.epochs}_val_log.pkl", "wb") as f: 
            pickle.dump(val_log, f)

    if args.eval_train_model: 
        with open("split_datasets_balanced_v2.pkl", "rb") as f: 
            split_datasets = pickle.load(f)
        test_dataloader = torch.utils.data.DataLoader(split_datasets[model_name]["test"], batch_size=8, shuffle=False)

        with open(f"{args.output_dir}/train/logs/{args.seed}/{model_name}_epoch10_val_log.pkl", "rb") as f: 
            val_log = pickle.load(f)
            
        # Select model by best validation accuracy 
        val_aucs = [x.item() for x in val_log['auc']]
        epochs_map = np.arange(2, 11, 2)
        checkpoint = str(epochs_map[np.argmax(np.array(val_aucs))])

        model_path = f"{args.output_dir}/train/models/{args.seed}/{model_name}_epoch{checkpoint}_model.pt"
        model = torch.load(model_path)
        
        print("Evaluate model")
        auc, task_aucs, task_outputs, task_targets = valid_test_epoch("test", 0, model, model_name, 
                                                                      "cuda", test_dataloader, 
                                                                      torch.nn.BCEWithLogitsLoss(), 
                                                                      num_classes=8,
                                                                     limit=limit)
        with open(f"{output_dir}/train/logs/{args.seed}/{model_name}_epoch{args.epochs}_auc.pkl", "wb") as f: 
            pickle.dump([auc], f)
        with open(f"{output_dir}/train/logs/{args.seed}/{model_name}_epoch{args.epochs}_task_aucs.pkl", "wb") as f: 
            pickle.dump(task_aucs, f)
        with open(f"{output_dir}/train/logs/{args.seed}/{model_name}_epoch{args.epochs}_task_outputs.pkl", "wb") as f: 
            pickle.dump(task_outputs, f)
        with open(f"{output_dir}/train/logs/{args.seed}/{model_name}_epoch{args.epochs}_task_targets.pkl", "wb") as f: 
            pickle.dump(task_targets, f)
        
    if args.eval_model:
        # Load preprocessed dataset
        with open("split_datasets_balanced_v2.pkl", "rb") as f: 
            split_datasets = pickle.load(f)
        if args.class_balanced:
            keyword = "balanced"
        else: 
            keyword = "match"
            
        test_dataloader = torch.utils.data.DataLoader(split_datasets[dataset_name]["test"], batch_size=8, shuffle=False)
        with open(f"{args.output_dir}/transfer/logs/{args.train_num}/{args.seed}/{model_name}_{dataset_name}_{keyword}_epoch10_val_log.pkl", "rb") as f: 
            val_log = pickle.load(f)
            
        # Select model by best validation accuracy 
        val_aucs = [x.item() for x in val_log['auc']]
        epochs_map = np.arange(2, 11, 2)
        checkpoint = str(epochs_map[np.argmax(np.array(val_aucs))])

        model_path = f"{args.output_dir}/transfer/models/{args.train_num}/{args.seed}/{model_name}_{dataset_name}_{keyword}_epoch{checkpoint}_model.pt"
        model = torch.load(model_path)
        
        # Evaluation 
        print("Evaluate model")
        auc, task_aucs, task_outputs, task_targets = valid_test_epoch("test", 0, model, model_name, 
                                                                      "cuda", test_dataloader, 
                                                                      torch.nn.BCEWithLogitsLoss(), 
                                                                      num_classes=8, 
                                                                      limit=limit)

        with open(f"{output_dir}/transfer/logs/{args.train_num}/{args.seed}/{model_name}_{dataset_name}_{keyword}_epoch{args.epochs}_auc.pkl", "wb") as f: 
            pickle.dump([auc], f) 
        with open(f"{output_dir}/transfer/logs/{args.train_num}/{args.seed}/{model_name}_{dataset_name}_{keyword}_epoch{args.epochs}_task_aucs.pkl", "wb") as f: 
            pickle.dump(task_aucs, f) 
        with open(f"{output_dir}/transfer/logs/{args.train_num}/{args.seed}/{model_name}_{dataset_name}_{keyword}_epoch{args.epochs}_task_outputs.pkl", "wb") as f: 
            pickle.dump(task_outputs, f) 
        with open(f"{output_dir}/transfer/logs/{args.train_num}/{args.seed}/{model_name}_{dataset_name}_{keyword}_epoch{args.epochs}_task_targets.pkl", "wb") as f: 
            pickle.dump(task_targets, f)   