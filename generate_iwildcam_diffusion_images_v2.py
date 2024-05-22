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
best_checkpoint_dreambooth = {
    "color-to-grayscale": "/mnt/scratch-lids/scratch/qixuanj/dreambooth/iwildcam_base_color-to-grayscale_checkpoint-2500/target/checkpoint-2000", 
    "grayscale-to-color": "/mnt/scratch-lids/scratch/qixuanj/dreambooth/iwildcam_base_grayscale-to-color_checkpoint-2000/target/checkpoint-2000", 
    "grayscale_day-to-night": "/mnt/scratch-lids/scratch/qixuanj/dreambooth/iwildcam_base_grayscale_day-to-night_checkpoint-2500/target/checkpoint-1500", 
    "grayscale_night-to-day": "/mnt/scratch-lids/scratch/qixuanj/dreambooth/iwildcam_base_grayscale_night-to-day_checkpoint-1500/target/checkpoint-1500"
}
# best_checkpoint_dreambooth = {
#     "color-to-grayscale": "/mnt/scratch-lids/scratch/qixuanj/dreambooth/iwildcam_base_color-to-grayscale_v2_checkpoint-1000/target/checkpoint-1500", 
#     "grayscale-to-color": "/mnt/scratch-lids/scratch/qixuanj/dreambooth/iwildcam_base_grayscale-to-color_v2_checkpoint-500/target/checkpoint-2000", 
#     "grayscale_day-to-night": "/mnt/scratch-lids/scratch/qixuanj/dreambooth/iwildcam_base_grayscale_day-to-night_v2_checkpoint-2000/target/checkpoint-1500", 
#     "grayscale_night-to-day": "/mnt/scratch-lids/scratch/qixuanj/dreambooth/iwildcam_base_grayscale_night-to-day_v2_checkpoint-500/target/checkpoint-1500"
# }
best_checkpoint_base = {
    "color-to-grayscale": "/mnt/scratch-lids/scratch/qixuanj/iwildcam_base_color-to-grayscale_v2/checkpoint-1000", 
    "grayscale-to-color": "/mnt/scratch-lids/scratch/qixuanj/iwildcam_base_grayscale-to-color_v2/checkpoint-500", 
    "grayscale_day-to-night": "/mnt/scratch-lids/scratch/qixuanj/iwildcam_base_grayscale_day-to-night_v2/checkpoint-500", 
    "grayscale_night-to-day": "/mnt/scratch-lids/scratch/qixuanj/iwildcam_base_grayscale_night-to-day_v2/checkpoint-500", 
}

parser = argparse.ArgumentParser(description='iWildCam image generation.')
parser.add_argument('--shift', type=str, default='color-to-grayscale')
args = parser.parse_args()

shift = args.shift

df_metadata = pd.read_csv("/mnt/scratch-lids/scratch/qixuanj/iwildcam_subset_organized/metadata_copy.csv", index_col=0)
# Segmentation model 
seg_pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

output_dir1 = f"/mnt/scratch-lids/scratch/qixuanj/iwildcam_generated_images_copy/{shift}/strength0.7"
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)
        
output_dir2 = f"/mnt/scratch-lids/scratch/qixuanj/iwildcam_generated_images_copy/{shift}/strength0.9"
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

output_dir3 = f"/mnt/scratch-lids/scratch/qixuanj/iwildcam_generated_images_copy/{shift}/strength1.0"
if not os.path.exists(output_dir3):
    os.makedirs(output_dir3)
    
subset = df_metadata[(df_metadata[shift] == 'train')]
print(f"{shift}: {len(subset)} Images")

file_prefix = best_checkpoint_dreambooth[shift]
unet = UNet2DConditionModel.from_pretrained(f"{file_prefix}/unet", torch_dtype=torch.bfloat16,)
pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        unet=unet,
        torch_dtype=torch.bfloat16,
        safety_checker=None,)
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()

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

    prompt = f"a camera trap photo of {animal} with target-domain"
    image1 = pipe(prompt=prompt,
         negative_prompt=''' pixelated, jpeg artifacts, cartoon, anime, geometric patterns,
                             artwork, cgi, illustration, painting, blurry''',
         image=img,
         mask_image=mask_mod,
         strength=0.7,
         guidance_scale=20,
         num_inference_steps=50,).images[0]
    
    image2 = pipe(prompt=prompt,
         negative_prompt=''' pixelated, jpeg artifacts, cartoon, anime, geometric patterns,
                             artwork, cgi, illustration, painting, blurry''',
         image=img,
         mask_image=mask_mod,
         strength=0.9,
         guidance_scale=20,
         num_inference_steps=50,).images[0]
    
    image3 = pipe(prompt=prompt,
         negative_prompt=''' pixelated, jpeg artifacts, cartoon, anime, geometric patterns,
                             artwork, cgi, illustration, painting, blurry''',
         image=img,
         mask_image=mask_mod,
         strength=1.,
         guidance_scale=20,
         num_inference_steps=50,).images[0]

    image1.save(output_dir1 + f"/img{index}_{animal}.jpg") 
    image2.save(output_dir2 + f"/img{index}_{animal}.jpg") 
    image3.save(output_dir3 + f"/img{index}_{animal}.jpg") 