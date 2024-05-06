from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline, UNet2DConditionModel
from torcheval.metrics import BinaryAUROC, BinaryAccuracy, MulticlassAccuracy
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm 
import numpy as np
import pandas as pd 
from glob import glob
import subprocess
import gc 
import shutil
import random
import skimage, torch, torchvision
from tqdm import tqdm
from torchvision.transforms import v2
import pickle

print("token 2")
keywords = [
    "waterbirds_finetune_sd_token2_816",
    # "waterbirds_finetune_sd_transfer_10",
    # "waterbirds_finetune_sd_transfer_50",
    # "waterbirds_finetune_sd_transfer_100",
    # "waterbirds_finetune_sd_transfer_500",
    # "waterbirds_finetune_sd_transfer_816",
    
           ]

transform = torchvision.transforms.Compose([v2.CenterCrop(224),
                                            v2.RandomHorizontalFlip(),
                                            v2.ToTensor(),
                                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                                            ])
sigmoid = torch.nn.Sigmoid()

with open("waterbirds_results/oracle_group/val_aucs.pkl", "rb") as f: 
    val_aucs = pickle.load(f)
val_aucs = [x.item() for x in val_aucs]
epochs_map = np.arange(10, 101, 10)
checkpoint = str(epochs_map[np.argmax(np.array(val_aucs))])

model = torch.load(f"waterbirds_results/oracle_group/checkpoint{checkpoint}.pt")
model.to("cuda")
model.eval()

model_id = "runwayml/stable-diffusion-v1-5"
checkpoints = np.arange(200, 2001, 200)
num_images = 10

for keyword in keywords:
    checkpoint_results = {}
    for checkpoint in checkpoints: 
        file_prefix = f"/mnt/scratch-lids/scratch/qixuanj/{keyword}/checkpoint-{checkpoint}"
        unet = UNet2DConditionModel.from_pretrained(f"{file_prefix}/unet")
        pipe = DiffusionPipeline.from_pretrained(model_id, unet=unet, dtype=torch.bfloat16, safety_checker=None,)
        pipe.to("cuda")
        
        # prompts = [
        #     "a photo of landbird with background of <wb-source-domain>", 
        #     "a photo of waterbird with background of <wb-source-domain>",
        #     "a photo of landbird with background of <wb-target-domain>", 
        #     "a photo of waterbird with background of <wb-target-domain>",
        # ]
        
        # prompts = [
        #     "a photo of ls", 
        #     "a photo of lt", 
        #     "a photo of ws", 
        #     "a photo of wt",
        # ]
        prompts = [
            "a photo of landbird-source", 
            "a photo of landbird-target", 
            "a photo of waterbird-target", 
            "a photot of waterbird-source",
        ]
        
    
        acc_results = []
        for i, p in enumerate(prompts): 
            images = pipe(prompt=p, 
                          strength=0.9, guidance_scale=7.5, num_inference_steps=50, 
                          num_images_per_prompt=num_images).images
            images_t = []
            for image in images: 
                images_t.append(transform(image).to("cuda"))
            images_t = torch.stack(images_t)
            outputs = model(images_t).detach().cpu().squeeze()
            outputs = sigmoid(outputs)
            # if "landbird" in p and "<wb-source-domain>" in p: 
            #     targets = torch.full((num_images, 1), 0).squeeze()
            # elif "landbird" in p and "<wb-target-domain>" in p: 
            #     targets = torch.full((num_images, 1), 1).squeeze()
            # elif "landbird" in p and "forest" in p: 
            #     targets = torch.full((num_images, 1), 0).squeeze()
            # elif "landbird" in p and "ocean" in p: 
            #     targets = torch.full((num_images, 1), 1).squeeze()
            # elif "waterbird" in p and "<wb-source-domain>" in p: 
            #     targets = torch.full((num_images, 1), 3).squeeze()
            # elif "waterbird" in p and "<wb-target-domain>" in p: 
            #     targets = torch.full((num_images, 1), 2).squeeze()
            # elif "waterbird" in p and "ocean" in p: 
            #     targets = torch.full((num_images, 1), 3).squeeze()
            # elif "waterbird" in p and "forest" in p: 
            #     targets = torch.full((num_images, 1), 2).squeeze()
            
            # if "ls" in p: 
            #     targets = torch.full((num_images, 1), 0).squeeze()
            # elif "lt" in p: 
            #     targets = torch.full((num_images, 1), 1).squeeze()
            # elif "ws" in p: 
            #     targets = torch.full((num_images, 1), 3).squeeze() 
            # elif "wt" in p: 
            #     targets = torch.full((num_images, 1), 2).squeeze() 

            if "landbird-source" in p: 
                targets = torch.full((num_images, 1), 0).squeeze()
            elif "landbird-target" in p: 
                targets = torch.full((num_images, 1), 1).squeeze()
            elif "waterbird-source" in p: 
                targets = torch.full((num_images, 1), 3).squeeze() 
            elif "waterbird-target" in p: 
                targets = torch.full((num_images, 1), 2).squeeze() 
    
            acc = MulticlassAccuracy()
            acc.update(outputs, targets)
            acc_results.append(acc.compute())
        print(acc_results)
        print(np.mean(np.array(acc_results)))
        checkpoint_results[checkpoint] = acc_results
    with open(f"/mnt/scratch-lids/scratch/qixuanj/{keyword}/checkpoint_group_results2.pkl", "wb") as f: 
        pickle.dump(checkpoint_results, f) 

    print(keyword)
    for checkpoint, result in checkpoint_results.items():
        result = [x.item() for x in result]
        print(f"checkpoint {checkpoint}: {round(np.mean(np.array(result)), 3)}")