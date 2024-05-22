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


_DESCRIPTION = "Stable Diffusion: ISIC source domain dataset with patches removed."

class ISIC_Dataset_Base(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
            datasets.BuilderConfig(name="benign", version=VERSION),
            datasets.BuilderConfig(name="malignant", version=VERSION),
            datasets.BuilderConfig(name="all", version=VERSION),
    ]
    
    DEFAULT_CONFIG_NAME = "all"
    
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
                                      "root": "/data/scratch/wgerych/spurious_ISIC_ruler_no_patches/", 
                                  }),
        ]
        
    def _generate_examples(self, split, root):
        train_ds = SpuriousDermDataset(file_path=root, split="train")
        val_ds = SpuriousDermDataset(file_path=root, split="val")
        # extra_dataset = SpuriousDermDataset(file_path=root, split='extra')
        # class1 = list(extra_dataset.metadata_for_split[extra_dataset.metadata_for_split['class'] == 1].sample(n=10, random_state=0).index)
        # class2 = list(extra_dataset.metadata_for_split[extra_dataset.metadata_for_split['class'] == 2].sample(n=10, random_state=0).index)
        # new_extra_dataset = torch.utils.data.Subset(extra_dataset, class1 + class2) 

        # Train training, validation, extra all at once 
        # split_data = torch.utils.data.ConcatDataset([train_ds, val_ds, new_extra_dataset])

        if self.config.name == "all: 
            split_data = torch.utils.data.ConcatDataset([train_ds, val_ds])
        elif self.config.name == "benign": 
            train_benign = np.where(train_ds.metadata_for_split['benign_malignant'] == 'benign')[0]
            val_benign = np.where(val_ds.metadata_for_split['benign_malignant'] == 'benign')[0]
            split_data = torch.utils.data.ConcatDataset([torch.utils.data.Subset(train_ds, train_benign), 
                                                         torch.utils.data.Subset(val_ds, val_benign)])
        elif self.config.name == "malignant": 
            train_malignant = np.where(train_ds.metadata_for_split['benign_malignant'] == 'malignant')[0]
            val_malignant = np.where(val_ds.metadata_for_split['benign_malignant'] == 'malignant')[0]
            split_data = torch.utils.data.ConcatDataset([torch.utils.data.Subset(train_ds, train_malignant), 
                                                            torch.utils.data.Subset(val_ds, val_malignant)])
                                                
        for idx in range(len(split_data)): 
            item = split_data[idx]
            img, y, group = item[0], item[1], item[2]
            # if group == 0: 
            #     token = "malignant-source"
            # elif group == 1: 
            #     token = "malignant-target" 
            # elif group == 2: 
            #     token = "benign-target" 
            # elif group == 3: 
            #     token = "benign-source"
            # else: 
            #     raise Exception("Unknown group label")

            if group == 0: 
                token = "malignant"
            elif group == 3: 
                token = "benign"
            else: 
                raise Exception("Unknown group label")

            prompt = f"a dermoscopic image of {token} skin lesion"
            
            yield idx, {
                "img": img,
                "prompt": prompt,
            }