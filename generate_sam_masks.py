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

from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from transformers import AutoImageProcessor, AutoModelForImageClassification

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.cm import get_cmap
from sklearn.metrics import silhouette_score
from glob import glob
import pickle


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

# Load ImageNet 
transform = transforms.Compose([
    transforms.ToTensor()
])
imagenet_data = torchvision.datasets.ImageNet('/data/healthy-ml/gobi1/data/ILSVRC2012/', split='val', transform = transform)


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint='/data/healthy-ml/scratch/qixuanj/generative_validation/sam_vit_h_4b8939.pth')
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam, 
                    points_per_side=15, # 32 
                    pred_iou_thresh=0.88, # 0.88 
                    stability_score_thresh=0.96, # 0.95
                    crop_n_layers=0,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100, # 0
                    box_nms_thresh= 0.7, #0.7
                    crop_nms_thresh= 0.7, # 0.7
                    )

image_encoder_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large-imagenet1k-1-layer')
image_encoder = AutoModelForImageClassification.from_pretrained('facebook/dinov2-large-imagenet1k-1-layer')

spurious_dir = "/data/healthy-ml/scratch/qixuanj/generative_validation/spurious_imagenet/dataset/spurious_imagenet/images"
subdirs = glob(spurious_dir + "/*/", recursive = True)

target_classes = []
for d in subdirs: 
    target_classes.append(int(d.split("/")[-2].split("class_")[1].split("_")[0]))
    
for target_class in tqdm(target_classes):
    target_indices = np.where(np.array(imagenet_data.targets) == target_class)[0]
    
    all_masks = {}
    all_centroids = {}
    all_colors = {}
    
    for i in target_indices:
        p, c = imagenet_data.imgs[i]
        img = cv2.imread(p)
        print(img.size)
        image_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        sam_result = mask_generator.generate(image_rgb)
        
        # Iterate through each SAM mask 
        all_logits = []
        for j in range(len(sam_result)):
            mask = np.repeat(sam_result[j]['segmentation'][:, :, np.newaxis], 3, axis=2)
            img_copy = img.copy()
            # Set region outside the segmented object to 0 
            img_copy[~mask] = 0
        
            # Run segmented image through encoder
            inputs = image_encoder_processor(images=img_copy, return_tensors="pt")
            outputs = image_encoder(**inputs)
            logits = outputs.logits
            # Save the logits 
            all_logits.append(logits.detach().numpy())
        all_logits = np.array(all_logits).squeeze()
    
        # Dimensionality reduction 
        pca = PCA(n_components=2)
        transformed_logits = pca.fit_transform(all_logits)
        
        # choose optimal K on original image encoder representation space 
        if len(transformed_logits) < 6: 
            optimal_k = len(transformed_logits)
            labels = list(range(optimal_k))
            centroids = all_logits
        else:
            max_cluster_num = min(max(16, int(len(transformed_logits) / 5 + 1)), int(len(transformed_logits)))
            min_cluster_num = max(5, int(len(transformed_logits) / 5))
            potential_k = range(min_cluster_num, max_cluster_num)
            sil = []
            for k in potential_k: 
                kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(all_logits)
                labels = kmeans.labels_
                sil.append(silhouette_score(all_logits, labels, metric = 'euclidean'))
            optimal_k = potential_k[np.argmax(sil)]
            kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init="auto").fit(all_logits)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
        
        masks = {}
        colors = {}
        for c in range(optimal_k): 
            mask = np.full(img.shape[:2], False)
            for j in np.where(labels == c)[0]: 
                mask = mask | sam_result[j]['segmentation']
            masks[c] = mask
            colors[c] = tuple(image_rgb[mask].mean(axis=0).astype(int))
        all_masks[p] = masks
        all_centroids[p] = centroids
        all_colors[p] = colors
        
    # Aggregate the logits together from the images 
    agg_logits = np.concatenate(tuple(all_centroids.values()))
    all_colors_list = []
    for i in range(len(all_colors)):
        for t in list(list(all_colors.values())[i].values()):
            all_colors_list.append(t)
    all_colors_list = np.array(all_colors_list)
    
    agg_logits_mod = np.concatenate([agg_logits, all_colors_list], axis=1)

    num_masks = []
    for m in all_masks.values(): 
        num_masks.append(len(m))
    print(np.array(num_masks).mean())
    
    optimal_k = int(np.array(num_masks).mean())
    kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init="auto").fit(agg_logits_mod)
    labels = kmeans.labels_

    grouped_masks = {}
    for i in range(optimal_k): 
        grouped_masks[i] = [[], []]
    
    counter = 0    
    for path, mask_dict in all_masks.items(): 
        for key, value in mask_dict.items():
            grouped_masks[labels[counter]][0].append(path)
            grouped_masks[labels[counter]][1].append(value)
            counter += 1
            
    savedir = f"spurious_imagenet/dataset/sam_masks_grouping/imagenet_class{target_class}"
    if not os.path.exists(savedir): 
        os.makedirs(savedir)

    savedir2 = f"spurious_imagenet/dataset/sam_raw_masks/imagenet_class{target_class}"
    if not os.path.exists(savedir2): 
        os.makedirs(savedir2)
        
    for i in range(optimal_k): 
        if not os.path.exists(savedir + f"/group{i}"): 
            os.makedirs(savedir + f"/group{i}")
        if not os.path.exists(savedir2 + f"/group{i}"): 
            os.makedirs(savedir2 + f"/group{i}")
            
        for j in range(len(grouped_masks[i][0])): 
            img = cv2.imread(grouped_masks[i][0][j])
            img_copy = img.copy()
            mask = grouped_masks[i][1][j]
            img_copy[~mask] = 0

            with open(savedir2 + f"/group{i}/mask{j}_" + grouped_masks[i][0][j].rsplit("/")[-1].split(".")[0] + ".pkl", "wb") as handle:
                pickle.dump(mask, handle)
    
            img_copy = Image.fromarray(img_copy)
            img_copy.save(savedir + f"/group{i}/mask{j}_" + grouped_masks[i][0][j].rsplit("/")[-1])