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
import random
import torchvision.datasets as dsets
from collections import Counter
import argparse 
from transformers import pipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import UNet2DConditionModel

data_dir = "/mnt/scratch-lids/scratch/qixuanj/iwildcam_subset_organized"
shift = 'grayscale_day-to-night'

df_metadata = pd.read_csv("/mnt/scratch-lids/scratch/qixuanj/iwildcam_subset_organized/metadata_copy.csv", index_col=0)
seg_pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

output_dir1 = f"/mnt/scratch-lids/scratch/qixuanj/iwildcam_generated_images_copy/{shift}/no_branches_strength0.9"
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)
        
output_dir2 = f"/mnt/scratch-lids/scratch/qixuanj/iwildcam_generated_images_copy/{shift}/branches_strength0.9"
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)
    
subset = df_metadata[(df_metadata[shift] == 'train')]
print(f"{shift}: {len(subset)} Images")

file_prefix = "/mnt/scratch-lids/scratch/qixuanj/dreambooth/iwildcam_base_grayscale_day-to-night_checkpoint-2500"
b_unet = UNet2DConditionModel.from_pretrained(f"{file_prefix}/branches_v2/checkpoint-1000/unet", torch_dtype=torch.bfloat16,)
b_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        unet=b_unet,
        torch_dtype=torch.bfloat16,
        safety_checker=None,)
b_pipe = b_pipe.to("cuda")
b_pipe.enable_model_cpu_offload()

nb_unet = UNet2DConditionModel.from_pretrained(f"{file_prefix}/no_branches_v3/checkpoint-1000/unet", torch_dtype=torch.bfloat16,)
nb_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        unet=nb_unet,
        torch_dtype=torch.bfloat16,
        safety_checker=None,)
nb_pipe = nb_pipe.to("cuda")
nb_pipe.enable_model_cpu_offload()

for i in range(len(subset)): 
    # Index correspond to the original metadata csv 
    index = subset.index[i]
    animal = subset['class_name'].iloc[i]
    
    img = Image.open(data_dir + "/" + subset['img_path'].iloc[i]).convert("RGB")

    if animal == 'background': 
        mask_mod = Image.fromarray(np.full(np.array(img).shape[:2], 255).astype(np.uint8)).convert("L")
    else: 
        pillow_mask = seg_pipe(img, return_mask = True)
        mask = np.array(pillow_mask)
        # Dilate to cover more area around animal
        mask = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=3)
        inv_mask = np.invert(mask)
        # Get mask of background and convert to PIL
        mask_mod = Image.fromarray(np.uint8(inv_mask)).convert("L")

    prompt = f"a camera trap photo with target-domain"
    image1 = nb_pipe(prompt=prompt,
         negative_prompt=''' pixelated, jpeg artifacts, cartoon, anime, geometric patterns,
                             artwork, cgi, illustration, painting, blurry''',
         image=img,
         mask_image=mask_mod,
         strength=0.9,
         guidance_scale=15,
         num_inference_steps=50,
         num_images_per_prompt=3)
    
    image2 = b_pipe(prompt=prompt,
         negative_prompt=''' pixelated, jpeg artifacts, cartoon, anime, geometric patterns,
                             artwork, cgi, illustration, painting, blurry''',
         image=img,
         mask_image=mask_mod,
         strength=0.9,
         guidance_scale=15,
         num_inference_steps=50,).images[0]
    
    for j, image in enumerate(image1.images): 
        image.save(output_dir1 + f"/image{index}_{j}_{animal}.png")
    image2.save(output_dir2 + f"/img{index}_{animal}.jpg")  