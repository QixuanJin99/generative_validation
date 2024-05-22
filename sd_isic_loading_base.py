import datasets
import cv2
from PIL import Image
import numpy as np 
import pandas as pd
import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms

class SpuriousDermDataset(VisionDataset):
    def __init__(self, file_path, split='train', transform=None):
        super(SpuriousDermDataset, self).__init__(root=None, transform=transform)

        self.file_path = file_path
        self.split = split
        self.transform = transform
        self.label_map = {'malignant': 1, 'benign': 0}

        # Load metadata from CSV
        self.metadata = pd.read_csv(file_path+'metadata.csv')

        # Filter metadata based on split
        self.metadata_for_split = self.metadata.iloc[[self.split in x for x in self.metadata['image']]].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata_for_split)

    def __getitem__(self, index):
        img_path = self.file_path + self.metadata_for_split.iloc[index]['image']
        melanoma_label = self.label_map[self.metadata_for_split.iloc[index]['benign_malignant']]
        group_label = self.metadata_for_split.iloc[index]['class']

        # Load image
        img = Image.open(img_path).convert('RGB')


        if self.transform:
            img = self.transform(img)
        else: 
            img = img.resize((512, 512))

        return img, melanoma_label, group_label


_DESCRIPTION = "Stable Diffusion: ISIC source domain dataset."

class ISIC_Dataset_Base(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")
    
    def _info(self): 
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
            {
                "img": datasets.Image(), 
                "prompt": datasets.Value("string"),
            })
        )
    
    def _split_generators(self, dl_manager): 
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, 
                                  gen_kwargs={
                                      "split" : "train",
                                      "root": "/data/scratch/wgerych/spurious_ISIC/", 
                                  }),
        ]
        
    def _generate_examples(self, split, root):
        train_ds = SpuriousDermDataset(file_path=root, split="train")
        prev_extra_dataset = SpuriousDermDataset(file_path=root, split="extra")
        groups = prev_extra_dataset.metadata_for_split['class']
        group1_indices = groups[groups == 1].iloc[:50].index
        group2_indices = groups[groups == 2].iloc[:50].index
        group1 = torch.utils.data.Subset(prev_extra_dataset, group1_indices)
        group2 = torch.utils.data.Subset(prev_extra_dataset, group2_indices)

        # Train training and extra all at once 
        split_data = torch.utils.data.ConcatDataset([train_ds, group1, group2])
        # split_data = torch.utils.data.ConcatDataset([group1, group2])
                                                
        for idx in range(len(split_data)): 
            item = split_data[idx]
            img, y, group = item[0], item[1], item[2]
            if group == 0: 
                token = "malignant-source"
            elif group == 1: 
                token = "malignant-target" 
            elif group == 2: 
                token = "benign-target" 
            elif group == 3: 
                token = "benign-source"
            else: 
                raise Exception("Unknown group label")

            prompt = f"a dermoscopic image of {token} skin lesion"
            
            yield idx, {
                "img": img,
                "prompt": prompt,
            }