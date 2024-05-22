import torch
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm 
import numpy as np
import pandas as pd 
from glob import glob
import random
import json
import PIL
import pickle
from torchvision.datasets import VisionDataset
from torchvision import transforms
import torchvision
from torchvision.transforms import v2
    
from isic_dataset import SpuriousDermDataset
from torcheval.metrics import MulticlassAUROC, MulticlassAccuracy, BinaryAUROC, BinaryAccuracy

epochs = 10
checkpoint_epochs = 2
output_dir = "/mnt/scratch-lids/scratch/qixuanj/isic_results_resnet50_no_patches/domain_seed0"

if not os.path.exists(output_dir): 
    os.makedirs(output_dir)

transform = torchvision.transforms.Compose([v2.Resize((224, 224)), 
                                            v2.RandomHorizontalFlip(),
                                            v2.ToImage(),
                                            v2.ToDtype(torch.float32, scale=True),
                                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                                            ])

filepath = "/data/scratch/wgerych/spurious_ISIC_ruler_no_patches/"
prev_train_dataset = SpuriousDermDataset(file_path=filepath, split='train', transform=transform)
prev_extra_dataset = SpuriousDermDataset(file_path=filepath, split='extra', transform=transform)
test_dataset = SpuriousDermDataset(file_path=filepath, split='test', transform=transform)

oracle_dataset = torch.utils.data.ConcatDataset([prev_train_dataset, prev_extra_dataset])
train_dataloader = torch.utils.data.DataLoader(oracle_dataset, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

model = torchvision.models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
model.classifier = torch.nn.Linear(1024, 1)
model.to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.BCELoss()

rulers_map = {
    0: 1, 
    1: 0, 
    2: 1, 
    3: 0,
}


train_aucs = [] 
test_aucs = []

avg_loss = []
for epoch in range(epochs):
    model.train()
    task_outputs=[]
    task_targets=[]
    
    t = tqdm(train_dataloader)
    for batch_idx, sample in enumerate(t): 
        optimizer.zero_grad()
        images, groups = sample[0], sample[2]
        rulers = np.vectorize(rulers_map.get)(groups)
        rulers = torch.Tensor(rulers).to("cuda")
        images = images.to("cuda")
        outputs = model(images)
        sigmoid = torch.nn.Sigmoid()
        outputs = sigmoid(outputs)
        outputs = outputs.squeeze(1)
        
        rulers = rulers.to(torch.float32)
        loss = criterion(outputs, rulers)
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.detach().cpu().numpy())
        task_outputs.append(outputs.detach().cpu().numpy())
        task_targets.append(rulers.detach().cpu().numpy())
        t.set_description(f'Epoch {epoch + 1} - Loss = {np.mean(avg_loss):4.4f}')
        
    task_outputs = np.concatenate(task_outputs)
    task_targets = np.concatenate(task_targets)
    auc = BinaryAUROC()
    auc.update(torch.Tensor(task_outputs), torch.Tensor(task_targets))
    auc_result = auc.compute()
    train_aucs.append(auc_result)
    
    if (epoch + 1) % checkpoint_epochs == 0: 
        print("checkpoint!")
        model.eval()
        t = tqdm(test_dataloader)
        test_task_outputs=[]
        test_task_targets=[]
        with torch.no_grad():
            for batch_idx, sample in enumerate(t):
                images, groups = sample[0], sample[2]
                rulers = np.vectorize(rulers_map.get)(groups)
                rulers = torch.Tensor(rulers).to("cuda")
                images = images.to("cuda")

                outputs = model(images)
                sigmoid = torch.nn.Sigmoid()
                outputs = sigmoid(outputs)
                outputs = outputs.squeeze(1)
                rulers = rulers.to(torch.torch.float32)
                
                test_task_outputs.append(outputs.detach().cpu().numpy())
                test_task_targets.append(rulers.detach().cpu().numpy())
        test_task_outputs = np.concatenate(test_task_outputs)
        test_task_targets = np.concatenate(test_task_targets)
        auc = BinaryAUROC()
        auc.update(torch.Tensor(test_task_outputs), torch.Tensor(test_task_targets))
        test_auc_result = auc.compute()
        test_aucs.append(test_auc_result)
        print(f'Test Epoch {epoch + 1} - Avg AUC = {test_auc_result:4.4f}')
        torch.save(model, f"{output_dir}/checkpoint{epoch + 1}.pt")
    print(f'Train Epoch {epoch + 1} - Avg AUC = {auc_result:4.4f}')

torch.save(model, f"{output_dir}/model.pt")
with open(f"{output_dir}/train_aucs.pkl", "wb") as f: 
    pickle.dump(train_aucs, f)  
with open(f"{output_dir}/test_aucs.pkl", "wb") as f: 
    pickle.dump(test_aucs, f)
