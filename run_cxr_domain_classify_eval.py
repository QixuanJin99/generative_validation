import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
import pickle 
import cv2
import torchxrayvision as xrv
import skimage, torch, torchvision
from tqdm import tqdm
from torchvision.transforms import v2
import os
from diffusers import DiffusionPipeline, UNet2DConditionModel
from torcheval.metrics import MulticlassAUROC, MulticlassAccuracy

print("set 3")
keywords = [
    # "cxr_transfer_sd_mimic_chexpert_1000_balanced",
    # "cxr_transfer_sd_mimic_padchest_1000_balanced", 
    # "cxr_transfer_sd_mimic_nih_1000_balanced",
    # "cxr_transfer_sd_mimic_chexpert_1000_match", 
    # "cxr_transfer_sd_mimic_padchest_1000_match", 
    # "cxr_transfer_sd_mimic_nih_1000_match", 
    # "cxr_transfer_sd_mimic_chexpert_500_match", 
    # "cxr_transfer_sd_mimic_chexpert_500_balanced", 
    # "cxr_transfer_sd_mimic_chexpert_100_match", 
    # "cxr_transfer_sd_mimic_chexpert_100_balanced", 
    # "cxr_transfer_sd_mimic_chexpert_50_match", 
    # "cxr_transfer_sd_mimic_chexpert_50_balanced",
    # "cxr_transfer_sd_mimic_chexpert_10_match", 
    # "cxr_transfer_sd_mimic_chexpert_10_balanced", 
]
num_images = 5
label_mapping = {'mimic': 0, 'chexpert': 1, 'padchest': 2, 'nih': 3}

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224),
                                            v2.ToDtype(torch.float32, scale=True),
                                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                                            ])
sigmoid = torch.nn.Sigmoid()

oracle = torch.load("cxr_domain_classify/model.pt")
oracle.eval()
oracle.to("cuda")

checkpoints = np.arange(200, 1001, 200)
model_id = "runwayml/stable-diffusion-v1-5"
condition_prompts = ["Cardiomegaly, Pneumonia", "No Findings", "", "Lesion"]

for keyword in keywords: 
    results = {}
    for checkpoint in checkpoints:
        results[checkpoint] = {}
        print("checkpoint {}".format(checkpoint))
        
        file_prefix = f"/mnt/scratch-lids/scratch/qixuanj/{keyword}/checkpoint-{checkpoint}"
        unet = UNet2DConditionModel.from_pretrained(f"{file_prefix}/unet")
        
        pipe = DiffusionPipeline.from_pretrained(model_id, unet=unet, dtype=torch.float16, safety_checker=None)
        pipe.to("cuda")
    
        source_dataset = keyword.split("_")[3]
        target_dataset = keyword.split("_")[4]
    
        # Source 
        all_labels = []
        all_outputs = []
        for i, condition in enumerate(condition_prompts): 
            prompt = f"a radiograph from dataset {source_dataset} with conditions {condition}"
            images = pipe(prompt=prompt, 
                          strength=0.9, guidance_scale=7.5, num_inference_steps=30, 
                          num_images_per_prompt=num_images).images
            images_t = []
            for image in images: 
                image = torch.Tensor(transform(np.array(image)))
                image = torch.cat((image, image, image), 0)
                images_t.append(image)
            images_t = torch.stack(images_t)
            images_t = images_t.to("cuda")
            
            with torch.no_grad():
                outputs = oracle(images_t)
            
            labels = np.repeat(label_mapping[source_dataset], num_images)
        
            all_outputs.append(outputs.detach().cpu().numpy())
            all_labels.append(labels)
        all_outputs = np.concatenate(all_outputs)
        all_labels = np.concatenate(all_labels)
        
        acc = MulticlassAccuracy()
        acc.update(torch.Tensor(all_outputs), torch.Tensor(all_labels))
        source_auc_result = acc.compute().item()
        
        # Target 
        all_labels = []
        all_outputs = []
        for i, condition in enumerate(condition_prompts): 
            prompt = f"a radiograph from dataset {target_dataset} with conditions {condition}"
            images = pipe(prompt=prompt, 
                          strength=0.9, guidance_scale=7.5, num_inference_steps=30, 
                          num_images_per_prompt=num_images).images
            images_t = []
            for image in images: 
                image = torch.Tensor(transform(np.array(image)))
                image = torch.cat((image, image, image), 0)
                images_t.append(image)
            images_t = torch.stack(images_t)
            images_t = images_t.to("cuda")
            
            with torch.no_grad():
                outputs = oracle(images_t)
            
            labels = np.repeat(label_mapping[target_dataset], num_images)
        
            all_outputs.append(outputs.detach().cpu().numpy())
            all_labels.append(labels)
        all_outputs = np.concatenate(all_outputs)
        all_labels = np.concatenate(all_labels)
    
        acc = MulticlassAccuracy()
        acc.update(torch.Tensor(all_outputs), torch.Tensor(all_labels))
        target_auc_result = acc.compute().item()
    
        results[checkpoint]['source_acc'] = source_auc_result
        results[checkpoint]['target_acc'] = target_auc_result
        print(source_auc_result) 
        print(target_auc_result)
        
    result_dir = f"/mnt/scratch-lids/scratch/qixuanj/{keyword}/eval"
    if not os.path.exists(result_dir): 
        os.makedirs(result_dir)
    with open(result_dir + "/checkpoint_oracle_acc.pkl", "wb") as f: 
        pickle.dump(results, f) 