import torch
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import supervision as sv
import skimage
from tqdm import tqdm 
import numpy as np
import torchvision
from torchvision import transforms
import pandas as pd
import gc

from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from transformers import pipeline
from transformers import AutoProcessor,AutoModel

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.cm import get_cmap
from sklearn.metrics import silhouette_score
from glob import glob
import pickle

import random
random.seed(0)

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


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint='/data/healthy-ml/scratch/qixuanj/generative_validation/sam_vit_h_4b8939.pth')
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam, 
                    points_per_side=128, # 32 
                    pred_iou_thresh=0.88, # 0.88 
                    stability_score_thresh=0.96, # 0.95
                    crop_n_layers=0,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100, # 0
                    box_nms_thresh= 0.7, #0.7
                    crop_nms_thresh= 0.7, # 0.7                      
                    )


#load the embedding model
model = AutoModel.from_pretrained("Idan0405/ClipMD",trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Idan0405/ClipMD")

metadata = pd.read_csv("/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG/mimic-cxr-2.0.0-metadata.csv.gz")
# Get only frontal images, either PA or AP 
meta = metadata[(metadata['ViewPosition'] == "PA") | (metadata['ViewPosition'] == "AP")]
meta['img_path'] = meta.apply(lambda row: 'p' + str(row['subject_id'])[:2] + '/p' + str(row['subject_id']) + \
                               '/s' + str(row['study_id']) + '/' + row['dicom_id'] + '.jpg', axis = 1)
labels = pd.read_csv("/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG/mimic-cxr-2.0.0-chexpert.csv.gz")
print("Total labels: {}".format(len(labels)))
support_device_labels = labels[labels['Support Devices'] == 1]
print("Num of support device labels: {}".format(len(support_device_labels)))
meta = meta[meta['study_id'].isin(support_device_labels['study_id'])]
print("Frontal support device labels: {}".format(len(meta)))

img_dir = "/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG/files_preprocessed_1024/"

offset = 0
num_images = 100
max_sam_masks = 30
optimal_k = 15

all_masks = {} 
embeds = [] 
for i in tqdm(range(offset, offset + num_images)):
    embed_list = []
    p = img_dir + meta['img_path'].iloc[i]
    img = cv2.imread(p)
    image_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    np_image = np.array(image_rgb)
    
    sam_result = mask_generator.generate(image_rgb)

    max_masks = min(max_sam_masks, len(sam_result))
    masks = [sam_result[j]['segmentation'] for j in range(max_masks)]
    all_masks[p] = masks

    #loop through the masks
    for _mask in masks:
        masked_image = np_image * _mask[:, :, np.newaxis] #mask out the image for the th image
        
        #embed the masked image
        inputs = processor(images=[masked_image], return_tensors="pt")
        image_features = model.get_image_features(**inputs).detach().to('cpu').numpy()
        embed_list.append(image_features)    
    embeds.append(embed_list)


del meta
del metadata
gc.collect()
    
# Aggregate the logits together from the images 
agg_logits = np.concatenate(tuple(embeds)).squeeze()

# PCA for KMeans 
pca = PCA(n_components=min(len(agg_logits), 64))
transformed_logits_cluster = pca.fit_transform(agg_logits)
    
kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init="auto").fit(transformed_logits_cluster)
labels = kmeans.labels_

grouped_masks = {}
for i in range(optimal_k): 
    grouped_masks[i] = [[], []]

counter = 0    
for path, mask_list in all_masks.items(): 
    for value in mask_list:
        grouped_masks[labels[counter]][0].append(path)
        grouped_masks[labels[counter]][1].append(value)
        counter += 1

batch_number = 1
savedir_base = f"/data/healthy-ml/scratch/qixuanj/generative_validation/mimic-cxr-results-batch{batch_number}-copy/"
savedir = f"{savedir_base}sam_masks_mimic-cxr_black"
if not os.path.exists(savedir): 
    os.makedirs(savedir)

savedir2 = f"{savedir_base}sam_masks_mimic-cxr_white"
if not os.path.exists(savedir2): 
    os.makedirs(savedir2)

savedir3 = f"{savedir_base}sam_masks_mimic-cxr_annotated"
if not os.path.exists(savedir3): 
    os.makedirs(savedir3)

savedir4 = f"{savedir_base}sam_masks_mimic-cxr_raw"
if not os.path.exists(savedir4): 
    os.makedirs(savedir4)
    
for i in range(optimal_k): 
    if not os.path.exists(savedir + f"/group{i}"): 
        os.makedirs(savedir + f"/group{i}")
    if not os.path.exists(savedir2 + f"/group{i}"): 
        os.makedirs(savedir2 + f"/group{i}")
    if not os.path.exists(savedir3 + f"/group{i}"): 
        os.makedirs(savedir3 + f"/group{i}")
    if not os.path.exists(savedir4 + f"/group{i}"): 
        os.makedirs(savedir4 + f"/group{i}")
        
    for j in range(len(grouped_masks[i][0])): 
        img = cv2.imread(grouped_masks[i][0][j])
        mask = grouped_masks[i][1][j]

        image_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        np_image = np.array(image_rgb)

        black_mask = np_image * mask[:, :, np.newaxis]
        
        mask_inv = np.invert(mask)
        white_mask = np_image.copy()
        white_mask[mask_inv] = [255, 255, 255]
    
        colored_img = np_image.copy()
        colored_img[mask] = [255, 0, 0]
        annotated_img = cv2.addWeighted(np_image, 0.6, colored_img, 0.4,0)

        partition = grouped_masks[i][0][j].split("/")[-3]
        study = grouped_masks[i][0][j].split("/")[-2]
        img_id = grouped_masks[i][0][j].split("/")[-1].split(".")[0]

        with open(savedir4 + f"/group{i}/batch{batch_number}_mask{j}_{partition}_{study}_{img_id}.pkl", "wb") as handle:
            pickle.dump(mask, handle)

        black_mask = Image.fromarray(black_mask)
        black_mask.save(savedir + f"/group{i}/batch{batch_number}_mask{j}_{partition}_{study}_{img_id}.jpg")

        white_mask = Image.fromarray(white_mask)
        white_mask.save(savedir2 + f"/group{i}/batch{batch_number}_mask{j}_{partition}_{study}_{img_id}.jpg")

        annotated_img = Image.fromarray(annotated_img)
        annotated_img.save(savedir3 + f"/group{i}/batch{batch_number}_mask{j}_{partition}_{study}_{img_id}.jpg")