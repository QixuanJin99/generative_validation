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
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics import MulticlassAUROC

class CXRDomain(Dataset):
    def __init__(self, file_path, split, transform=None):
        self.file_path = file_path 
        self.split = split 
        self.transform = transform 
        with open(self.file_path, "rb") as f: 
            prompt_files = pickle.load(f) 
        df = pd.concat([prompt_files['mimic'][split], prompt_files['chexpert'][split], 
                        prompt_files['padchest'][split], prompt_files['nih'][split]]).reset_index(drop=True)
        self.label_mapping = {'mimic': 0, 'chexpert': 1, 'padchest': 2, 'nih': 3}
        df['dataset_label'] = df['dataset_name'].map(self.label_mapping)
        self.df = df
        
    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx): 
        img_dirs = {"mimic": "/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG/files", 
                    "padchest": "/data/healthy-ml/gobi1/data/PadChest/images-224", 
                    "chexpert": "/data/healthy-ml/gobi1/data", 
                    "nih": "/data/healthy-ml/gobi1/data/ChestXray8/images"}
        
        sample = self.df.iloc[idx]
        img = cv2.imread(img_dirs[sample['dataset_name']] + "/" + sample['file_suffix'])
        if self.transform: 
            img = self.transform(img) 
        return {'img': img, 'label': sample['dataset_label']}

output_dir = "/data/healthy-ml/scratch/qixuanj/generative_validation/cxr_domain_classify"
if not os.path.exists(output_dir): 
    os.makedirs(output_dir)

model = torchvision.models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
model.classifier = torch.nn.Linear(1024, 4)

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224),
                                            v2.ToDtype(torch.float32, scale=True),
                                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                                            ])
train_ds = CXRDomain("cxr_prompt_files_base10000.pkl", "train", transform)
test_ds = CXRDomain("cxr_prompt_files_base10000.pkl", "test", transform)
train_dataloader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=16, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Training loop 
epochs = 10 
checkpoint_epochs = 2

model.to("cuda")
model.train()
avg_loss = []
for epoch in range(epochs): 
    task_outputs=[]
    task_targets=[]
    
    t = tqdm(train_dataloader)
    for batch_idx, sample in enumerate(t): 
        optimizer.zero_grad()
        images, targets = sample['img'], sample['label']
        images = torch.cat((images, images, images), 1)
        images = images.to("cuda")
        targets = targets.to("cuda")
        outputs = model(images)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.detach().cpu().numpy())
        task_outputs.append(outputs.detach().cpu().numpy())
        task_targets.append(targets.detach().cpu().numpy())
        t.set_description(f'Epoch {epoch + 1} - Loss = {np.mean(avg_loss):4.4f}')
        
    task_outputs = np.concatenate(task_outputs)
    task_targets = np.concatenate(task_targets)
    auc = MulticlassAUROC(num_classes=4)
    auc.update(torch.Tensor(task_outputs), torch.Tensor(task_targets))
    auc_result = auc.compute()

    if epoch % checkpoint_epochs: 
        torch.save(model, f"{output_dir}/checkpoint{epoch}.pt")
    print(f'Epoch {epoch + 1} - Avg AUC = {auc_result:4.4f}')
torch.save(model, f"{output_dir}/model.pt")
