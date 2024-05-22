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

parser = argparse.ArgumentParser(description='ISIC image generation.')
parser.add_argument('--datadir', type=str, default="/data/scratch/wgerych/spurious_ISIC_ruler_no_patches/")
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=1039)
args = parser.parse_args()

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
keyword = "isic_sd_base_no_patches_checkpoint-2000"

malignant_file_prefix = f"{base_dir}/dreambooth/{keyword}/malignant-source/checkpoint-500"
malignant_unet = UNet2DConditionModel.from_pretrained(f"{malignant_file_prefix}/unet")
malignant_pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, unet=malignant_unet, dtype=torch.bfloat16, safety_checker=None,)
malignant_pipe.to("cuda")

benign_file_prefix = f"{base_dir}/dreambooth/{keyword}/benign-source/checkpoint-500"
benign_unet = UNet2DConditionModel.from_pretrained(f"{benign_file_prefix}/unet")
benign_pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, unet=benign_unet, dtype=torch.bfloat16, safety_checker=None,)
benign_pipe.to("cuda")

generator1 = torch.Generator("cuda").manual_seed(0)

output_dir2 = base_dir + "/isic_generated_images/" + keyword + "/source"
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

max_idx = min(len(train_dataset), end_idx)
for i in tqdm(range(start_idx, max_idx)):
    img, class_label, group_label = train_dataset[i]

    mask = train_dataset_mask[i][0]
    mask_mod = PIL.ImageOps.invert(mask)

    # Cross generation
    if group_label == 0:
        token = "malignant"
        prompt = f"a dermoscopic image of benign-source skin lesion"
        image2 = benign_pipe(prompt=prompt,
                         negative_prompt=''' pixelated, blurry, jpeg artifacts, low quality,
                                             cartoon, artwork, cgi, illustration, painting, overexposed,
                                             grayscale, grainy, white spots, multiple angles, ''',
                         image=img,
                         mask_image=mask_mod,
                         strength=0.7,
                         guidance_scale=15,
                         num_inference_steps=50,
                         generator=generator1,).images[0]
    elif group_label == 3:
        token = "benign"
        prompt = f"a dermoscopic image of malignant-source skin lesion"
        image2 = malignant_pipe(prompt=prompt,
                         negative_prompt=''' pixelated, blurry, jpeg artifacts, low quality,
                                             cartoon, artwork, cgi, illustration, painting, overexposed,
                                             grayscale, grainy, white spots, multiple angles, ''',
                         image=img,
                         mask_image=mask_mod,
                         strength=0.7,
                         guidance_scale=15,
                         num_inference_steps=50,
                         generator=generator1,).images[0]
    else: 
        raise Exception("Target images included")

    image2.save(output_dir2 + f"/image{i}_{token}.png")