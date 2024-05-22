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

parser = argparse.ArgumentParser(description='Waterbirds image generation.')
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=1000)
args = parser.parse_args()

start_idx = args.start_idx
end_idx = args.end_idx

with open("waterbirds_preprocessed_datasets_classifier_pil.pkl", "rb") as f: 
    preprocessed_datasets = pickle.load(f)
train_ds = preprocessed_datasets['train']

from transformers import pipeline
seg_pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

from diffusers import StableDiffusionInpaintPipeline
from diffusers import UNet2DConditionModel

base_dir = "/mnt/scratch-lids/scratch/qixuanj"
model_id = "runwayml/stable-diffusion-v1-5"

landbird_file_prefix = f"{base_dir}/dreambooth/waterbirds_finetune_sd_token2_816_checkpoint-1600/landbird-target100/checkpoint-1500"
landbird_unet = UNet2DConditionModel.from_pretrained(f"{landbird_file_prefix}/unet")
landbird_pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, unet=landbird_unet, dtype=torch.bfloat16, safety_checker=None,)
landbird_pipe.to("cuda")

waterbird_file_prefix = f"{base_dir}/dreambooth/waterbirds_finetune_sd_token2_816_checkpoint-1600/waterbird-target100/checkpoint-2000"
waterbird_unet = UNet2DConditionModel.from_pretrained(f"{waterbird_file_prefix}/unet")
waterbird_pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, unet=waterbird_unet, dtype=torch.bfloat16, safety_checker=None,)
waterbird_pipe.to("cuda")

# Load textual inversion
# tx_inv_dir = "/mnt/scratch-lids/scratch/qixuanj/textual_inversion/waterbirds_finetune_sd_token2_816_checkpoint-1600"
# pipe.load_textual_inversion(tx_inv_dir + "/landbird-source/learned_embeds-steps-500.safetensors", "landbird-source")
# pipe.load_textual_inversion(tx_inv_dir + "/landbird-target/learned_embeds-steps-500.safetensors", "landbird-target")
# pipe.load_textual_inversion(tx_inv_dir + "/waterbird-target/learned_embeds-steps-2000.safetensors", "waterbird-target")
# pipe.load_textual_inversion(tx_inv_dir + "/waterbird-source/learned_embeds-steps-2000.safetensors", "waterbird-source") 

# pipe.to("cuda")

generator1 = torch.Generator("cuda").manual_seed(0)
# generator2 = torch.Generator("cuda").manual_seed(1)

keyword = "waterbirds_finetune_sd_token2_816_dreambooth"
output_dir1 = base_dir + "/waterbird_generated_images/" + keyword + "/mix_strength"
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)
# output_dir2 = base_dir + "/waterbird_generated_images/" + keyword + "/strength0.7"
# if not os.path.exists(output_dir2):
#     os.makedirs(output_dir2)

for i in tqdm(range(start_idx, end_idx)):
    img, class_label, group_label = train_ds[i]
    img = img.convert("RGB")
    pillow_mask = seg_pipe(img, return_mask = True)
    mask = np.array(pillow_mask)
    inv_mask = np.invert(mask)
    # Get mask of background and convert to PIL
    mask_mod = Image.fromarray(np.uint8(inv_mask)).convert("L")

    if group_label == 0:
        prompt = f"a photo of landbird-target"
        image1 = landbird_pipe(prompt=prompt,
                     negative_prompt=''' two birds, pixelated, blurry, jpeg artifacts, low quality,
                                         cartoon, artwork, cgi, illustration, painting, overexposed,
                                         grayscale, grainy, white spots, multiple angles, ''',
                     image=img,
                     mask_image=mask_mod,
                     strength=0.7,
                     guidance_scale=20,
                     num_inference_steps=50,
                     generator=generator1,).images[0]
    elif group_label == 3:
        prompt = f"a photo of waterbird-target"
        image1 = waterbird_pipe(prompt=prompt,
                     negative_prompt=''' two birds, pixelated, blurry, jpeg artifacts, low quality,
                                         cartoon, artwork, cgi, illustration, painting, overexposed,
                                         grayscale, grainy, white spots, multiple angles, ''',
                     image=img,
                     mask_image=mask_mod,
                     strength=0.5,
                     guidance_scale=20,
                     num_inference_steps=50,
                     generator=generator1,).images[0]
    else: 
        raise Exception("Target images included")

    # prompt = f"a photo of {token}"
    # image1 = pipe(prompt=prompt,
    #              negative_prompt=''' two birds, pixelated, blurry, jpeg artifacts, low quality,
    #                                  cartoon, artwork, cgi, illustration, painting, overexposed,
    #                                  grayscale, grainy, white spots, multiple angles, ''',
    #              image=img,
    #              mask_image=mask_mod,
    #              strength=0.7,
    #              guidance_scale=20,
    #              num_inference_steps=50,
    #              generator=generator1,).images[0]
    # image2 = pipe(prompt=prompt,
    #              negative_prompt=''' two birds, pixelated, blurry, jpeg artifacts, low quality,
    #                                  cartoon, artwork, cgi, illustration, painting, overexposed,
    #                                  grayscale, grainy, white spots, multiple angles, ''',
    #              image=img,
    #              mask_image=mask_mod,
    #              strength=0.9,
    #              guidance_scale=20,
    #              num_inference_steps=50,
    #              generator=generator2,).images[0]

    image1.save(output_dir1 + f"/image{i}.png")
    # image2.save(output_dir2 + f"/image{i}.png")