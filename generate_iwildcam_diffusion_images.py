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
    "color-to-grayscale": "/mnt/scratch-lids/scratch/qixuanj/dreambooth/iwildcam_base_color-to-grayscale_checkpoint-2500", 
    "grayscale-to-color": "/mnt/scratch-lids/scratch/qixuanj/dreambooth/iwildcam_base_grayscale-to-color_checkpoint-2000", 
    "grayscale_day-to-night": "/mnt/scratch-lids/scratch/qixuanj/dreambooth/iwildcam_base_grayscale_day-to-night_checkpoint-2500", 
    "grayscale_night-to-day": "/mnt/scratch-lids/scratch/qixuanj/dreambooth/iwildcam_base_grayscale_night-to-day_checkpoint-1500"
}
best_checkpoint_base = {
    "color-to-grayscale": "/mnt/scratch-lids/scratch/qixuanj/iwildcam_base_color-to-grayscale/checkpoint-2500", 
    "grayscale-to-color": "/mnt/scratch-lids/scratch/qixuanj/iwildcam_base_grayscale-to-color/checkpoint-2000", 
    "grayscale_day-to-night": "/mnt/scratch-lids/scratch/qixuanj/iwildcam_base_grayscale_day-to-night/checkpoint-2500", 
    "grayscale_night-to-day": "/mnt/scratch-lids/scratch/qixuanj/iwildcam_base_grayscale_night-to-day/checkpoint-1500", 
}
best_token_checkpoints = {
    "color-to-grayscale": {
        "cattle_target": 1000,
        "dik-diks_target": 2000, 
        "elephants_target": 1000, 
        "giraffes_target": 1500,
        "impalas_target": 500, 
        "zebras_target": 500,
    }, 
    "grayscale-to-color": {
        "cattle_target": 1500,
        "dik-diks_target": 1500, 
        "elephants_target": 1000,
        "giraffes_target": 1500, 
        "impalas_target": 2000, 
        "zebras_target": 1000,
    }, 
    "grayscale_day-to-night": {
        "background_target": 1500,
        "dik-diks_target": 500, 
        "elephants_target": 1500, 
        "giraffes_target": 1500, 
        "impalas_target": 1000,
    }, 
    "grayscale_night-to-day": {
        "cattle_target": 2000, 
        "dik-diks_target": 500, 
        "elephants_target": 1000, 
        "giraffes_target": 1000, 
        "impalas_target": 2000,
        "zebras_target": 1000,
    }, 
}
all_tokens = ["background_target", "cattle_target", "dik-diks_target", "elephants_target", 
             "giraffes_target", "impalas_target",  "zebras_target"]

parser = argparse.ArgumentParser(description='iWildCam image generation.')
parser.add_argument('--shift', type=str, default='color-to-grayscale')
args = parser.parse_args()

shift = args.shift

generator1 = torch.Generator("cuda").manual_seed(0)
generator2 = torch.Generator("cuda").manual_seed(1)

df_metadata = pd.read_csv("/mnt/scratch-lids/scratch/qixuanj/iwildcam_subset_organized/metadata.csv", index_col=0)
# Segmentation model 
seg_pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

output_dir1 = f"/mnt/scratch-lids/scratch/qixuanj/iwildcam_generated_images/{shift}/strength0.7"
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)
        
output_dir2 = f"/mnt/scratch-lids/scratch/qixuanj/iwildcam_generated_images/{shift}/strength0.9"
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)
    
for token in all_tokens: 
    animal = token.split("_")[0]
    domain = token.split("_")[1]
    subset = df_metadata[(df_metadata[shift] == 'train') & (df_metadata['class_name'] == animal)]
    print(f"{shift} {token}: {len(subset)} Images")
    
    if token in best_token_checkpoints[shift]: 
        file_prefix = best_checkpoint_dreambooth[shift] + f"/{token}/checkpoint-{best_token_checkpoints[shift][token]}"
    else: 
        # Use base model if dreambooth model is not available 
        file_prefix = best_checkpoint_base[shift]
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
        
        img = Image.open(data_dir + "/" + subset['img_path'].iloc[i]).convert("RGB")
        pillow_mask = seg_pipe(img, return_mask = True)
        mask = np.array(pillow_mask)
        # Dilate to cover more area around animal
        mask = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=3)
        inv_mask = np.invert(mask)
        # Get mask of background and convert to PIL
        mask_mod = Image.fromarray(np.uint8(inv_mask)).convert("L")

        prompt = f"a camera trap photo of {animal} with {domain}-domain"
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

        image1.save(output_dir1 + f"/img{index}_{animal}.jpg") 
        image2.save(output_dir2 + f"/img{index}_{animal}.jpg") 