import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
import pickle 
import cv2
import torchxrayvision as xrv
import skimage, torch, torchvision
from sklearn.model_selection import GroupShuffleSplit
import os
from tqdm import tqdm
import argparse

from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
from diffusers import UNet2DConditionModel

from CheXmask_Database.DataPostprocessing.utils import get_mask_from_RLE

from skimage.metrics import structural_similarity

parser = argparse.ArgumentParser(description='Generate Stable Diffusion images')
parser.add_argument('--source_dataset', type=str, default="")
args = parser.parse_args()

model_path = "/data/scratch/qixuanj/cxr_finetune_sd"
file_prefix = "/data/scratch/qixuanj/cxr_finetune_sd/checkpoint-3000"
    
output_dir = "/data/healthy-ml/scratch/qixuanj/generative_validation/data/cxr_finetune_sd/checkpoint-3000"
seed = 0
max_samples = 10000
offset = 0
# max_samples = 8069
# offset = 1031
strengths = [0.2, 0.5, 0.8]

# Current only support running one of the below flags 
run_inpainting = True
run_img2img = False

if args.source_dataset == "mimic": 
    run_id = "source_mimic"
    experiments = [
        ('mimic', 'padchest'),
         ('mimic', 'chexpert'),
         ('mimic', 'nih'),
    ]
elif args.source_dataset == "chexpert":
    run_id = "source_chexpert"
    experiments = [
        ('chexpert', 'mimic'),
         ('chexpert', 'padchest'),
         ('chexpert', 'nih'),
    ]
elif args.source_dataset == "padchest":
    run_id = "source_padchest"
    experiments = [
        ('padchest', 'mimic'),
         ('padchest', 'chexpert'),
         ('padchest', 'nih'),
    ]
elif args.source_dataset == "nih":
    run_id = "source_nih"
    experiments = [
        ('nih', 'mimic'),
         ('nih', 'padchest'),
         ('nih', 'chexpert'),
    ]
else: 
    raise Exception("Source dataset not valid")

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
generator2 = torch.Generator("cuda").manual_seed(seed + 100)

pathologies = np.array(["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
               "Lesion", "Pneumonia", "Pneumothorax", "No Finding"])

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(512)])
d_chex = xrv.datasets.CheX_Dataset(imgpath="/data/healthy-ml/gobi1/data/CheXpert-v1.0-small",
                                   csvpath="/data/healthy-ml/gobi1/data/CheXpert-v1.0-small/train.csv",
                                   transform=transform, views=["PA", "AP"], unique_patients=False)
d_pc = xrv.datasets.PC_Dataset(imgpath="/data/healthy-ml/gobi1/data/PadChest/images-224", 
                               transform=transform, views=["PA", "AP"], unique_patients=False)
d_mimic = xrv.datasets.MIMIC_Dataset(imgpath="/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG/files", 
                                    csvpath="/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG/mimic-cxr-2.0.0-chexpert.csv.gz", 
                                    metacsvpath="/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG/mimic-cxr-2.0.0-metadata.csv.gz", 
                                    transform=transform, views=["AP", "PA"], unique_patients=False)
d_nih = xrv.datasets.NIH_Dataset(imgpath="/data/healthy-ml/gobi1/data/ChestXray8/images",
                                 transform=transform, views=["PA","AP"], unique_patients=False)
# Standardization for NIH 
new_labels = pd.DataFrame(d_nih.labels, columns = d_nih.pathologies)
# Combine "Mass" and "Nodule" as "Lesion" class
new_labels['Lesion'] = new_labels['Mass'] + new_labels['Nodule']
new_labels['Lesion'][new_labels['Lesion'] > 1] = 1
# If all negative findings for original labels, set as no finding  
new_labels['No Finding'] = new_labels.eq(0).all(axis=1).astype(float)
d_nih.pathologies = new_labels.columns.values
d_nih.labels = new_labels.values
xrv.datasets.relabel_dataset(pathologies, d_nih)

# Standardization for PadChest 
new_labels = pd.DataFrame(d_pc.labels, columns = d_pc.pathologies) 
# Combine "Mass" and "Nodule" as "Lesion" class
new_labels['Lesion'] = new_labels['Mass'] + new_labels['Nodule']
new_labels['Lesion'][new_labels['Lesion'] > 1] = 1
no_findings_list = list(new_labels.columns)
no_findings_list.remove("Support Devices") 
no_findings_list.remove("Tube") 
print(no_findings_list)

# If all negative findings for selected labels, set as no finding  
new_labels['No Finding'] = new_labels[no_findings_list].eq(0).all(axis=1).astype(float)
d_pc.pathologies = new_labels.columns.values
d_pc.labels = new_labels.values
xrv.datasets.relabel_dataset(pathologies, d_pc)

# Standardization for CheXpert 
d_chex.pathologies = ["Lesion" if x == "Lung Lesion" else x for x in d_chex.pathologies]
new_labels = pd.DataFrame(d_chex.labels, columns = d_chex.pathologies) 
no_findings_list = list(new_labels.columns)
no_findings_list.remove("Support Devices") 
# If all negative findings for selected labels, set as no finding  
new_labels['No Finding'] = new_labels[no_findings_list].eq(0).all(axis=1).astype(float)
d_chex.pathologies = new_labels.columns.values
d_chex.labels = new_labels.values
xrv.datasets.relabel_dataset(pathologies, d_chex)

# Standardization for MIMIC-CXR 
d_mimic.pathologies = ["Lesion" if x == "Lung Lesion" else x for x in d_mimic.pathologies]
new_labels = pd.DataFrame(d_mimic.labels, columns = d_mimic.pathologies) 
no_findings_list = list(new_labels.columns)
no_findings_list.remove("Support Devices") 
print(no_findings_list)
# If all negative findings for selected labels, set as no finding  
new_labels['No Finding'] = new_labels[no_findings_list].eq(0).all(axis=1).astype(float)
d_mimic.pathologies = new_labels.columns.values
d_mimic.labels = new_labels.values
xrv.datasets.relabel_dataset(pathologies, d_mimic)

original_datasets = {"mimic": d_mimic, 
                     "chexpert": d_chex, 
                     "nih": d_nih, 
                     "padchest": d_pc} 
split_datasets = {}
gss = GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=seed)
gss_val = GroupShuffleSplit(train_size=0.875,test_size=0.125, random_state=seed)

for name, dataset in original_datasets.items(): 
    train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
    train_inds, val_inds = next(gss_val.split(X=range(len(train_inds)), groups=train_inds))

    split_datasets[name] = {}
    split_datasets[name]["train"] = xrv.datasets.SubsetDataset(dataset, train_inds)
    # Generate samples for specific training section 
    max_total = min(offset + max_samples, len(train_inds))
    split_datasets[name]["train_subset"] = xrv.datasets.SubsetDataset(dataset, train_inds[offset:max_total])

# Load all mask dataframes necessary 
mask_dfs = {}
for exp in experiments: 
    source = exp[0]
    if source not in mask_dfs:
        mask_dfs[source] = pd.read_csv(mask_paths[source][1])

unet = UNet2DConditionModel.from_pretrained(f"{file_prefix}/unet", 
                                                    torch_dtype=torch.bfloat16,)
if run_inpainting: 
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        unet=unet,
        torch_dtype=torch.bfloat16,
    )

    pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()
if run_img2img: 
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_path,
                unet=unet,
                torch_dtype=torch.bfloat16,
            )

    pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()


ssim = {}
for exp in experiments: 
    source = exp[0]
    target = exp[1]
    print(f"converting source {source} to target {target}")
    if run_inpainting:
        for strength in strengths:
            if not os.path.exists(output_dir + f"/inpaint/{source}/{target}/strength{strength}/"): 
                os.makedirs(output_dir + f"/inpaint/{source}/{target}/strength{strength}")
    if run_img2img: 
        for strength in strengths: 
            if not os.path.exists(output_dir + f"/img2img/{source}/{target}/strength{strength}/"): 
                os.makedirs(output_dir + f"/img2img/{source}/{target}/strength{strength}")
    
    ssim[(source, target)] = {}
        
    for i in tqdm(range(len(split_datasets[source]["train_subset"]))): 
        _, label, img = split_datasets[source]["train_subset"][i].values()
        img = img.squeeze()
        img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
        img = Image.fromarray(img).convert("RGB")

        conditions = ", ".join(pathologies[np.argwhere(label == 1).squeeze()])
        
        if source == "chexpert": 
            uniq_id = split_datasets[source]['train_subset'].csv.iloc[i][mask_paths[source][0]].replace("CheXpert-v1.0-small/", "")
        else: 
            uniq_id = split_datasets[source]['train_subset'].csv.iloc[i][mask_paths[source][0]]
        example = mask_dfs[source][mask_dfs[source][mask_paths[source][0]] == uniq_id]
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

            ssim[(source, target)] = []
            for strength in strengths: 
                image = pipe(prompt=new_prompt, 
                             image=img, 
                             mask_image=mask_mod, 
                             strength=strength,
                             guidance_scale=7.5,
                             num_inference_steps=50, 
                             generator=generator).images[0]
                # If image is black, try again with second generator
                if np.array(image).mean() == 0: 
                    image = pipe(prompt=new_prompt, 
                             image=img, 
                             mask_image=mask_mod, 
                             strength=strength,
                             guidance_scale=7.5,
                             num_inference_steps=50, 
                             generator=generator2).images[0]

                image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
                (score, diff) = structural_similarity(img_gray, image_gray, full=True)
                # save similarity score
                ssim[(source, target)].append((strength, score)) 

                if source == "chexpert": 
                    image_id = uniq_id.split("/")[1] + "_" + uniq_id.split("/")[2] + "_" + uniq_id.split("/")[3].split(".")[0] 
                elif source == "mimic": 
                    image_id = uniq_id 
                else: 
                    image_id = uniq_id.split(".")[0]
                # Save image 
                image.save(output_dir + f"/inpaint/{source}/{target}/strength{strength}/{image_id}.png")
        if run_img2img: 
            img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
            new_prompt = f"a radiograph from dataset {target} with conditions {conditions}"

            ssim[(source, target)] = []
            for strength in strengths: 
                image = pipe(prompt=new_prompt, 
                             image=img,  
                             strength=strength,
                             guidance_scale=7.5,
                             num_inference_steps=50, 
                             generator=generator).images[0]
                # If image is black, try again with second generator
                if np.array(image).mean() == 0:
                    image = pipe(prompt=new_prompt, 
                             image=img,  
                             strength=strength,
                             guidance_scale=7.5,
                             num_inference_steps=50, 
                             generator=generator2).images[0]

                image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
                (score, diff) = structural_similarity(img_gray, image_gray, full=True)
                # save similarity score
                ssim[(source, target)].append((strength, score)) 

                if source == "chexpert": 
                    image_id = uniq_id.split("/")[1] + "_" + uniq_id.split("/")[2] + "_" + uniq_id.split("/")[3].split(".")[0] 
                elif source == "mimic": 
                    image_id = uniq_id 
                else: 
                    image_id = uniq_id.split(".")[0]
                # Save image 
                image.save(output_dir + f"/img2img/{source}/{target}/strength{strength}/{image_id}.png")

if run_inpainting:
    with open(output_dir + f"/inpaint/ssim_{run_id}.pkl", "wb") as f: 
        pickle.dump(ssim, f)
if run_img2img: 
    with open(output_dir + f"/img2img/ssim_{run_id}.pkl", "wb") as f: 
        pickle.dump(ssim, f)
print("Script Finished!")