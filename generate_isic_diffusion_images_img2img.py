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

parser = argparse.ArgumentParser(description='ISIC image generation (Img2Img).')
parser.add_argument('--datadir', type=str, default="/data/scratch/wgerych/spurious_ISIC_ruler_no_patches/")
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=1039)
args = parser.parse_args()

num_images = 5

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
# train_dataset_mask = torch.utils.data.ConcatDataset([torch.utils.data.Subset(prev_val_dataset_mask, add_indices),
#                                                     prev_train_dataset_mask])


# from diffusers import StableDiffusionPipeline
from diffusers import AutoPipelineForImage2Image
from diffusers import UNet2DConditionModel

base_dir = "/mnt/scratch-lids/scratch/qixuanj"
model_id = "runwayml/stable-diffusion-v1-5"
keyword = "isic_sd_base_no_patches_checkpoint-2000_target100"

model_id = "runwayml/stable-diffusion-v1-5"
file_prefix = f"/mnt/scratch-lids/scratch/qixuanj/dreambooth/isic_sd_base_no_patches_checkpoint-2000/target100/checkpoint-2500"
unet = UNet2DConditionModel.from_pretrained(f"{file_prefix}/unet")
pipe = AutoPipelineForImage2Image.from_pretrained(model_id, unet=unet, dtype=torch.bfloat16, safety_checker=None,)
pipe.to("cuda")

output_dir2 = base_dir + "/isic_generated_images/" + keyword + "/img2img/strength0.5"
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

label_map = {
    1: "malignant", 
    0: "benign",
}

label_offset = 0

max_idx = min(len(train_dataset), end_idx)
for i in range(start_idx, max_idx):
    print(f"Index {i}")
    img, class_label, group_label = train_dataset[i]
    token = label_map[class_label]

    # mask = train_dataset_mask[i][0]
    # mask_mod = PIL.ImageOps.invert(mask)

    prompt = f"a dermoscopic image of {token}-target skin lesion"
    image2 = pipe(prompt=prompt,
                  image=img,
                     negative_prompt=''' pixelated, blurry, jpeg artifacts, low quality,
                                         cartoon, artwork, cgi, illustration, painting, overexposed,
                                         grayscale, grainy, white spots, multiple angles, ''',
                     strength=0.5,
                     guidance_scale=15,
                     num_inference_steps=50,
                     num_images_per_prompt=num_images)
    for j, image in enumerate(image2.images): 
        image.save(output_dir2 + f"/image{i}_{label_offset + j}_{token}.png")