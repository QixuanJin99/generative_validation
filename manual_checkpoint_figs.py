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

from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline, UNet2DConditionModel

print("set 4")
keywords = [
            # "cxr_transfer_sd_mimic_chexpert_10_match", 
            # "cxr_transfer_sd_mimic_chexpert_500_balanced",  
            # "cxr_transfer_sd_mimic_chexpert_50_match", 
            # "cxr_transfer_sd_mimic_nih_1000_balanced", 
            # "cxr_transfer_sd_mimic_nih_1000_match", 
            # "cxr_transfer_sd_mimic_padchest_1000_balanced", 
            # "cxr_transfer_sd_mimic_padchest_1000_match",
    
            # "cxr_transfer_sd_mimic_chexpert_10_balanced", 
            # "cxr_transfer_sd_mimic_chexpert_50_balanced", 
            # "cxr_transfer_sd_mimic_chexpert_100_balanced", 
            # "cxr_transfer_sd_mimic_chexpert_500_balanced", 
            # "cxr_transfer_sd_mimic_chexpert_500_match",
]

keywords = [
    "cxr_finetune_sd_padchest_base"
]

model_id = "runwayml/stable-diffusion-v1-5"
# checkpoints = np.arange(200, 1001, 200)
checkpoints = np.arange(500, 5001, 500)


for keyword in keywords: 
    print(keyword)
    for checkpoint in checkpoints: 
        file_prefix = f"/mnt/scratch-lids/scratch/qixuanj/{keyword}/checkpoint-{checkpoint}"
        fig_dir = f"/mnt/scratch-lids/scratch/qixuanj/{keyword}/figs"
        if not os.path.exists(fig_dir): 
            os.makedirs(fig_dir)
        unet = UNet2DConditionModel.from_pretrained(f"{file_prefix}/unet")
        
        pipe = DiffusionPipeline.from_pretrained(model_id, unet=unet, dtype=torch.bfloat16)
        pipe.to("cuda")
        
        num_images = 5
        prompts = [
            "Cardiomegaly, Pneumonia", 
            "Consolidation", 
            "Lesion",
            "No Finding",
        ]
        dataset = keyword.split("_")[3]
        
        fig, ax = plt.subplots(len(prompts), num_images, figsize=(10,10))
        for i, p in enumerate(prompts): 
            prompt = f"a radiograph from dataset {dataset} with conditions {p}"
            images = pipe(prompt=prompt, 
                          strength=0.9, guidance_scale=7.5, num_inference_steps=50, 
                          num_images_per_prompt=num_images).images
            for j, img in enumerate(images): 
                ax[i, j].imshow(img)
                ax[i, j].set_xticklabels([])
                ax[i, j].set_yticklabels([])
            if p == "Cardiomegaly, Pneumonia": 
                p = "Cardiomegaly, \n Pneumonia"
            ax[i, 0].set_ylabel("dataset \n{}".format(p), rotation=0, labelpad=50)
        fig.tight_layout()
        plt.savefig(fig_dir + f"/checkpoint{checkpoint}.png", dpi=300)
        plt.show()
    

        
# for keyword in keywords: 
#     print(keyword)
#     for checkpoint in checkpoints: 
#         file_prefix = f"/mnt/scratch-lids/scratch/qixuanj/{keyword}/checkpoint-{checkpoint}"
#         fig_dir = f"/mnt/scratch-lids/scratch/qixuanj/{keyword}/figs"
#         if not os.path.exists(fig_dir): 
#             os.makedirs(fig_dir)
#         unet = UNet2DConditionModel.from_pretrained(f"{file_prefix}/unet")
        
#         pipe = DiffusionPipeline.from_pretrained(model_id, unet=unet, dtype=torch.bfloat16)
#         pipe.to("cuda")
        
#         num_images = 3
#         prompts = [
#             "Cardiomegaly, Pneumonia", 
#             "Consolidation", 
#             "Lesion",
#             "No Finding",
#         ]
        
#         fig, ax = plt.subplots(len(prompts), num_images, figsize=(10,10))
#         for i, p in enumerate(prompts): 
#             prompt = f"a radiograph from dataset mimic with conditions {p}"
#             images = pipe(prompt=prompt, 
#                           strength=0.9, guidance_scale=7.5, num_inference_steps=50, 
#                           num_images_per_prompt=num_images).images
#             for j, img in enumerate(images): 
#                 ax[i, j].imshow(img)
#                 ax[i, j].set_xticklabels([])
#                 ax[i, j].set_yticklabels([])
#             if p == "Cardiomegaly, Pneumonia": 
#                 p = "Cardiomegaly, \n Pneumonia"
#             ax[i, 0].set_ylabel("dataset \n{}".format(p), rotation=0, labelpad=50)
#         fig.tight_layout()
#         plt.savefig(fig_dir + f"/mimic_checkpoint{checkpoint}.png", dpi=300)
#         plt.show()
    
#         transfer_dataset = keyword.split("_")[-3]
#         fig, ax = plt.subplots(len(prompts), num_images, figsize=(10,10))
#         for i, p in enumerate(prompts): 
#             prompt = f"a radiograph from dataset {transfer_dataset} with conditions {p}"
#             images = pipe(prompt=prompt, 
#                           strength=0.9, guidance_scale=7.5, num_inference_steps=50, 
#                           num_images_per_prompt=num_images).images
#             for j, img in enumerate(images): 
#                 ax[i, j].imshow(img)
#                 ax[i, j].set_xticklabels([])
#                 ax[i, j].set_yticklabels([])
#             if p == "Cardiomegaly, Pneumonia": 
#                 p = "Cardiomegaly, \n Pneumonia"
#             ax[i, 0].set_ylabel("dataset \n{}".format(p), rotation=0, labelpad=50)
#         fig.tight_layout()
#         plt.savefig(fig_dir + f"/{transfer_dataset}_checkpoint{checkpoint}.png", dpi=300)
#         plt.show()