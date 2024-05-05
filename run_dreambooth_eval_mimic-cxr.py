import torch
import os
import cv2
from PIL import Image
from tqdm import tqdm 
import numpy as np
import pandas as pd 
from glob import glob
import subprocess
import gc 
import shutil

from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel

checkpoint_names = [
                    # "L-whole_L", 
                    # "L-circle_L", "R-whole_R", 
                    "metal-leads_lead", "pacemaker_pacemaker", "portable-left_portable"
                   ]
checkpoint_nums = ["500", "1000", "1500", "2000", "2500"]

dataset_name = "mimic-cxr"
for checkpoint_name in checkpoint_names: 
    feature_name = checkpoint_name.split("_")[0]
    for checkpoint_num in checkpoint_nums: 
        print(f"{checkpoint_name} with checkpoint {checkpoint_num}")
        file_prefix = f"/data/scratch/qixuanj/mimic-cxr_roentgen_dreambooth_ckpts/{checkpoint_name}/checkpoint-{checkpoint_num}"

        if not os.path.exists(file_prefix): 
            print("Model checkpoint does not exist!") 
            print(file_prefix)
            continue

        unet = UNet2DConditionModel.from_pretrained(f"{file_prefix}/unet")
        text_encoder = CLIPTextModel.from_pretrained(f"{file_prefix}/text_encoder")
        
        model_id = "/data/healthy-ml/scratch/qixuanj/generative_validation/roentgen" 
        pipe = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16)
        pipe.to("cuda")

        num_images = 10
        # Keep running model to remove black images from NSFW trigger 
        images = pipe(prompt=f"a photo of <{dataset_name}-{feature_name}>>", 
                      negative_prompt="",
                      strength=0.9, guidance_scale=7.5, num_inference_steps=50, 
                      num_images_per_prompt=num_images).images
        
        max_tries = 15
        prev_num_empty = 0
        while max_tries > 0: 
            num_empty = 0
            for img in images: 
                if np.array(img).mean() == 0: 
                    num_empty += 1
                    images.remove(img)
            # Also break out of loop if keep generating same number of black images 
            if num_empty == prev_num_empty: 
                break 
            else: 
                prev_num_empty = num_empty
        
            # No missing images to generate 
            if num_empty == 0: 
                break
            images += pipe(prompt=f"a photo of <{dataset_name}-{feature_name}>", 
                      negative_prompt="",
                      strength=0.9, guidance_scale=7.5, num_inference_steps=50, 
                      num_images_per_prompt=num_empty).images
            max_tries -= 1
        for img in images: 
            if np.array(img).mean() == 0: 
                num_empty += 1
                images.remove(img)
        print(f"Max tries {max_tries} left out of 15; Total {len(images)} images")

        output_dir = f"roentgen_mimic-cxr_dreambooth_imgs/{checkpoint_name}/checkpoint-{checkpoint_num}" 
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir) 

        for i, image in enumerate(images):
            image.save(output_dir + f"/image{i}.png")
            
        # Clean up memory 
        del unet
        del text_encoder
        del pipe
        gc.collect()