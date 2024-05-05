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
from transformers import MobileViTImageProcessor, MobileViTV2ForImageClassification

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.cm import get_cmap
from sklearn.metrics import silhouette_score
from glob import glob
import pickle

import networkx as nx
from networkx.algorithms.components.connected import connected_components

def to_graph(l):
    G = nx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """ 
        treat `l` as a Graph and returns it's edges 
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current


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

spurious_dir = "/data/healthy-ml/scratch/qixuanj/generative_validation/spurious_imagenet/dataset/spurious_imagenet/images"
subdirs = glob(spurious_dir + "/*/", recursive = True)

target_classes = []
for d in subdirs: 
    target_classes.append(int(d.split("/")[-2].split("class_")[1].split("_")[0]))

for target_class in tqdm(target_classes):
    target_indices = np.where(np.array(imagenet_data.targets) == target_class)[0]
    
    all_masks = {}
    all_logits = {}
    
    for i in target_indices:
        p, c = imagenet_data.imgs[i]
        img = cv2.imread(p)
        image_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        sam_result = mask_generator.generate(image_rgb)
        
        intersect_matrix = np.zeros((len(sam_result), len(sam_result)))                      
        for j in range(len(sam_result)):
            mask = np.repeat(sam_result[j]['segmentation'][:, :, np.newaxis], 3, axis=2)
            img_copy1 = img.copy()
            img_copy1[~mask] = 0
            hist_img1 = cv2.calcHist([img_copy1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
            hist_img1[0, 0, 0] = 0 
            hist_img1 = cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            for k in range(j + 1, len(sam_result)):
                mask = np.repeat(sam_result[k]['segmentation'][:, :, np.newaxis], 3, axis=2)
                img_copy2 = img.copy()
                img_copy2[~mask] = 0
                hist_img2 = cv2.calcHist([img_copy2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
                hist_img2[0, 0, 0] = 0 
                hist_img2 = cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
                intersect_matrix[j, k] =  cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_INTERSECT)
        intersect_matrix2 = intersect_matrix.copy()
        intersect_matrix2[intersect_matrix2 == 0] = np.nan
        intersect_thresholds = []
        
        for q in np.linspace(0.9, 0.95, 5):
            intersect_thresholds.append(np.nanquantile(intersect_matrix2, q = q))
        for intersect_threshold in intersect_thresholds:
            intersect_matrix2 = intersect_matrix.copy()
            intersect_matrix2[intersect_matrix2 < intersect_threshold] = 0
            pairs_list = list(zip(np.nonzero(intersect_matrix2)[0], np.nonzero(intersect_matrix2)[1]))
        
            G = to_graph(pairs_list)
            pattern = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
            pattern = [list(s) for s in pattern]
            if len(pattern) > 1: 
                break
        
        masks = []
        for pa in pattern: 
            mask = np.full(img.shape[:2], False)
            for idx in pa: 
                mask = mask | sam_result[idx]['segmentation']
            masks.append(mask)

        # Add the top 3 predicted_iou SAM segmentation masks  
        topn = min(len(sam_result), 3)
        for idx in range(topn):
            mask = np.full(img.shape[:2], False)
            mask = mask | sam_result[idx]['segmentation']
            masks.append(mask)
        
        stability_scores = []
        for i in range(len(sam_result)): 
            stability_scores.append(sam_result[i]['stability_score'])
        topn_idx = np.argpartition(np.array(stability_scores), -topn)[-topn:]

        # Add the top 3 stability_score SAM segmentation masks
        for idx in topn_idx: 
            mask = np.full(img.shape[:2], False)
            mask = mask | sam_result[idx]['segmentation']
            masks.append(mask)
    
        mask_imgs = []
        for m in masks: 
            mask = np.repeat(m[:, :, np.newaxis], 3, axis=2)
            img_copy = image_rgb.copy()
            img_copy[~mask] = 0
            mask_imgs.append(img_copy)

        logits_preds = []
        for m in mask_imgs: 
            inputs = feature_extractor(images=m, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            logits_preds.append(logits.detach().cpu())
            
        all_logits[p] = np.concatenate(tuple(logits_preds))
        all_masks[p] = masks
        
    # Aggregate the logits together from the images 
    agg_logits = np.concatenate(tuple(all_logits.values()))

    # PCA for KMeans 
    pca = PCA(n_components=64)
    transformed_logits_cluster = pca.fit_transform(agg_logits)
    
    sil = []
    potential_k = range(5, 16)
    for k in potential_k: 
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(transformed_logits_cluster)
        labels = kmeans.labels_
        sil.append(silhouette_score(transformed_logits_cluster, labels, metric = 'euclidean'))
    optimal_k = potential_k[np.argmax(sil)]
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
            
    savedir = f"/data/healthy-ml/scratch/qixuanj/generative_validation/spurious_imagenet/dataset/sam_masks_grouping_histogram/imagenet_class{target_class}"
    if not os.path.exists(savedir): 
        os.makedirs(savedir)

    savedir2 = f"/data/healthy-ml/scratch/qixuanj/generative_validation/spurious_imagenet/dataset/sam_raw_masks_histogram/imagenet_class{target_class}"
    if not os.path.exists(savedir2): 
        os.makedirs(savedir2)
        
    for i in range(optimal_k): 
        if not os.path.exists(savedir + f"/group{i}"): 
            os.makedirs(savedir + f"/group{i}")
        if not os.path.exists(savedir2 + f"/group{i}"): 
            os.makedirs(savedir2 + f"/group{i}")
            
        for j in range(len(grouped_masks[i][0])): 
            img = cv2.imread(grouped_masks[i][0][j])
            img_copy = img_rgb.copy()
            
            mask = grouped_masks[i][1][j]
            img_copy[~mask] = 0

            with open(savedir2 + f"/group{i}/mask{j}_" + grouped_masks[i][0][j].rsplit("/")[-1].split(".")[0] + ".pkl", "wb") as handle:
                pickle.dump(mask, handle)
    
            img_copy = Image.fromarray(img_copy)
            img_copy.save(savedir + f"/group{i}/mask{j}_" + grouped_masks[i][0][j].rsplit("/")[-1])