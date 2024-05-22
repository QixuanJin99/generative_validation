import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
import pickle 
import cv2
import torchxrayvision as xrv
import skimage, torch, torchvision
import os
from tqdm import tqdm
import argparse

from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
from diffusers import UNet2DConditionModel

from CheXmask_Database.DataPostprocessing.utils import get_mask_from_RLE

parser = argparse.ArgumentParser(description='Generate Stable Diffusion images')
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=9561)
parser.add_argument('--model_checkpoint_path', type=str, default="")
parser.add_argument('--strength', type=float, default=0.7)
parser.add_argument('--target_dataset', type=str, default="chexpert")
parser.add_argument('--style', type=str, default="inpainting")
args = parser.parse_args()

strength = args.strength
file_prefix = args.model_checkpoint_path
output_dir = f"/mnt/scratch-lids/scratch/qixuanj/cxr_generated_images/{args.target_dataset}"

seed = 0

# Current only support running one of the below flags
run_inpainting = False 
run_img2img = False

if args.style == "inpainting": 
    run_inpainting = True
else: 
    run_img2img = True

# location of preprocessed CheXMask 
mask_paths = {"mimic": ("dicom_id", "/data/healthy-ml/gobi1/data/chexmask-cxr-segmentation-data/0.2/Preprocessed/MIMIC-CXR-JPG.csv"), 
              "chexpert": ("Path", "/data/healthy-ml/gobi1/data/chexmask-cxr-segmentation-data/0.2/Preprocessed/CheXpert.csv"), 
              "padchest": ("ImageID", "/data/healthy-ml/gobi1/data/chexmask-cxr-segmentation-data/0.2/Preprocessed/Padchest.csv"), 
              "nih": ("Image Index", "/data/healthy-ml/gobi1/data/chexmask-cxr-segmentation-data/0.2/OriginalResolution/ChestX-Ray8.csv")}

def get_devices(gpus):
    if len(gpus) == 0:
        device_ids = None
        device = torch.device('cpu')
        print('Warning! Computing on CPU')
    elif len(gpus) == 1:
        device_ids = None
        device = torch.device('cuda:' + str(gpus[0]))
    else:
        device_ids = [int(i) for i in gpus]
        device = torch.device('cuda:' + str(min(device_ids)))
    return device, device_ids
device, device_ids = get_devices([0])

generator = torch.Generator("cuda").manual_seed(seed)

pathologies = np.array(["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
               "Lesion", "Pneumonia", "Pneumothorax", "No Finding"])

with open("cxr_prompt_files_base10000.pkl", "rb") as f: 
    metadata = pickle.load(f)
metadata = metadata['mimic']['train']
print("Source dataset with {} images".format(len(metadata)))
mimic_base_dir = "/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG/files"

mask_df = pd.read_csv(mask_paths['mimic'][1])

unet = UNet2DConditionModel.from_pretrained(f"{file_prefix}/unet", 
                                                    torch_dtype=torch.bfloat16,)
if run_inpainting: 
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        unet=unet,
        torch_dtype=torch.bfloat16,
        safety_checker=None,
    )

    pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()
if run_img2img: 
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                unet=unet,
                torch_dtype=torch.bfloat16,
                safety_checker=None,
            )
    pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()

source = "mimic"
target = args.target_dataset

print(f"converting source {source} to target {target}")
if run_inpainting:
    if not os.path.exists(output_dir + f"/inpaint/{source}/{target}/strength{strength}/"): 
        os.makedirs(output_dir + f"/inpaint/{source}/{target}/strength{strength}")
if run_img2img: 
    if not os.path.exists(output_dir + f"/img2img/{source}/{target}/strength{strength}/"): 
        os.makedirs(output_dir + f"/img2img/{source}/{target}/strength{strength}")

max_idx = min(args.end_idx, len(metadata))
for i in tqdm(range(args.start_idx, max_idx)):
    img_path = mimic_base_dir + "/" + metadata['file_suffix'].iloc[i]
    conditions = metadata['labels'].iloc[i]
    img = Image.open(img_path).convert("RGB")

    uniq_id = img_path.split("/")[-1].split(".")[0]
    example = mask_df[mask_df[mask_paths[source][0]] == uniq_id]
    if len(example) == 0: 
        continue
    height = example["Height"].iloc[0]
    width = example["Width"].iloc[0]
    rightLungMask_RLE = example["Right Lung"].iloc[0]
    leftLungMask_RLE = example["Left Lung"].iloc[0]
    heartMask_RLE = example["Heart"].iloc[0]

    rightLungMask = get_mask_from_RLE(rightLungMask_RLE, height, width)
    leftLungMask = get_mask_from_RLE(leftLungMask_RLE, height, width)
    heartMask = get_mask_from_RLE(heartMask_RLE, height, width)

    rightLungMask_resized = cv2.resize(rightLungMask, (512, 512), interpolation =cv2.INTER_NEAREST)
    leftLungMask_resized = cv2.resize(leftLungMask, (512, 512), interpolation =cv2.INTER_NEAREST)
    heartMask_resized = cv2.resize(heartMask, (512, 512), interpolation =cv2.INTER_NEAREST)
    combined_mask = rightLungMask_resized | leftLungMask_resized | heartMask_resized
    inv_mask = np.invert(combined_mask)
    mask_mod = Image.fromarray(np.uint8(inv_mask)).convert("L")
    
    if run_inpainting: 
        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        new_prompt = f"a radiograph from dataset {target} with conditions {conditions}"
        
 
        image = pipe(prompt=new_prompt, 
                     image=img, 
                     mask_image=mask_mod, 
                     strength=strength,
                     guidance_scale=15,
                     num_inference_steps=50, 
                     generator=generator).images[0]

        if source == "chexpert": 
            image_id = uniq_id.split("/")[1] + "_" + uniq_id.split("/")[2] + "_" + uniq_id.split("/")[3].split(".")[0] 
        elif source == "mimic": 
            image_id = uniq_id 
        else: 
            image_id = uniq_id.split(".")[0]
        # Save image 
        image.save(output_dir + f"/inpaint/{source}/{target}/strength{strength}/{image_id}_img{i}.png")
    if run_img2img: 
        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        new_prompt = f"a radiograph from dataset {target} with conditions {conditions}"

        image = pipe(prompt=new_prompt, 
                     image=img,  
                     strength=strength,
                     guidance_scale=15,
                     num_inference_steps=50, 
                     generator=generator).images[0]

        if source == "chexpert": 
            image_id = uniq_id.split("/")[1] + "_" + uniq_id.split("/")[2] + "_" + uniq_id.split("/")[3].split(".")[0] 
        elif source == "mimic": 
            image_id = uniq_id 
        else: 
            image_id = uniq_id.split(".")[0]
        # Save image 
        image.save(output_dir + f"/img2img/{source}/{target}/strength{strength}/{image_id}_img{i}.png")
print("Script Finished!")