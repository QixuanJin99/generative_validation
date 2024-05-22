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
import argparse 
import PIL
from isic_dataset import SpuriousDermDataset
from glob import glob

parser = argparse.ArgumentParser(description='ISIC image generation.')
parser.add_argument('--datadir', type=str, default="/data/scratch/wgerych/spurious_ISIC_ruler_no_patches/")
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=1039)
args = parser.parse_args()

num_background = 5

start_idx = args.start_idx
end_idx = args.end_idx

filepath = args.datadir
prev_train_dataset = SpuriousDermDataset(file_path=filepath, split='train')
prev_train_dataset_mask = SpuriousDermDataset(file_path=filepath, split='train', get_mask=True)

prev_val_dataset = SpuriousDermDataset(file_path=filepath, split='val')
prev_val_dataset_mask = SpuriousDermDataset(file_path=filepath, split='val', get_mask=True)

add_indices = list(prev_val_dataset.metadata_for_split.sample(n=len(prev_val_dataset) - 50, random_state=0).index)

train_dataset = torch.utils.data.ConcatDataset([torch.utils.data.Subset(prev_val_dataset, add_indices),
                                                    prev_train_dataset])
train_dataset_mask = torch.utils.data.ConcatDataset([torch.utils.data.Subset(prev_val_dataset_mask, add_indices),
                                                    prev_train_dataset_mask])


from diffusers import StableDiffusionInpaintPipeline
from diffusers import UNet2DConditionModel

base_dir = "/mnt/scratch-lids/scratch/qixuanj"
model_id = "runwayml/stable-diffusion-v1-5"
keyword = "isic_sd_base_no_patches_checkpoint-2000_target100background_blur"

file_prefix = f"/mnt/scratch-lids/scratch/qixuanj/dreambooth/isic_sd_base_no_patches_checkpoint-2000/target100/checkpoint-2500"
unet = UNet2DConditionModel.from_pretrained(f"{file_prefix}/unet")
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, unet=unet, dtype=torch.bfloat16, safety_checker=None,)
pipe.to("cuda")

output_dir = base_dir + f"/isic_generated_images/{keyword}/strength0.5"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_dir2 = base_dir + f"/isic_generated_images/{keyword}/strength0.7"
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

output_dir3 = base_dir + f"/isic_generated_images/{keyword}/strength0.3"
if not os.path.exists(output_dir3):
    os.makedirs(output_dir3)

background_images = glob("/mnt/scratch-lids/scratch/qixuanj/inversion_datasets/isic/target100background/*.jpg")

label_map = {
    1: "malignant", 
    0: "benign",
}

max_idx = min(len(train_dataset), end_idx)
for i in tqdm(range(start_idx, max_idx)):
    img, class_label, group_label = train_dataset[i]
    token = label_map[class_label]

    mask = train_dataset_mask[i][0]
    mask_mod = PIL.ImageOps.invert(mask)
    blurred_mask = pipe.mask_processor.blur(mask_mod, blur_factor=33)

    background_paths = random.sample(background_images, 5)

    for j in range(num_background): 
        background_image = Image.open(background_paths[j]).resize((512, 512))
        img_mod = PIL.Image.composite(img, background_image, mask)
        
        prompt = f"a dermoscopic image of {token}-target skin lesion"
        image = pipe(prompt=prompt,
                         negative_prompt=''' pixelated, blurry, jpeg artifacts, low quality,
                                             cartoon, artwork, cgi, illustration, painting, overexposed,
                                             grayscale, grainy, white spots, multiple angles, ''',
                         image=img_mod,
                         mask_image=blurred_mask,
                         strength=0.5,
                         guidance_scale=7.5,
                         num_inference_steps=50,).images[0]
        image2 = pipe(prompt=prompt,
                         negative_prompt=''' pixelated, blurry, jpeg artifacts, low quality,
                                             cartoon, artwork, cgi, illustration, painting, overexposed,
                                             grayscale, grainy, white spots, multiple angles, ''',
                         image=img_mod,
                         mask_image=blurred_mask,
                         strength=0.7,
                         guidance_scale=7.5,
                         num_inference_steps=50).images[0]
        image3 = pipe(prompt=prompt,
                         negative_prompt=''' pixelated, blurry, jpeg artifacts, low quality,
                                             cartoon, artwork, cgi, illustration, painting, overexposed,
                                             grayscale, grainy, white spots, multiple angles, ''',
                         image=img_mod,
                         mask_image=blurred_mask,
                         strength=0.3,
                         guidance_scale=7.5,
                         num_inference_steps=50).images[0]
        
        image.save(output_dir + f"/image{i}_{j}_{token}.png")
        image2.save(output_dir2 + f"/image{i}_{j}_{token}.png")
        image3.save(output_dir3 + f"/image{i}_{j}_{token}.png")
        
        # for j, image in enumerate(image.images): 
        #     image.save(output_dir + f"/image{i}_{j}_{token}.png")
        # for j, image in enumerate(image2.images): 
        #     image.save(output_dir2 + f"/image{i}_{j}_{token}.png")
            