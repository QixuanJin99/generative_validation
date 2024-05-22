import datasets
import cv2
from PIL import Image
import numpy as np 
import pandas as pd
import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms

class iWildCamDataset(VisionDataset):
    def __init__(self, file_path, shift, split='train', transform=None, version='v2'):
        super(iWildCamDataset, self).__init__(root=None, transform=transform)
        self.file_path = file_path
        self.shift = shift
        self.shift_names = ['color_day-to-night', 'grayscale_day-to-night', 'color_night-to-day',
                            'grayscale_night-to-day', 'color-to-grayscale', 'grayscale-to-color']
        
        self.split = split
        self.transform = transform
        self.label_to_class_name = {0: 'background',
                                     1: 'cattle',
                                     2: 'elephants',
                                     3: 'impalas',
                                     4: 'zebras',
                                     5: 'giraffes',
                                     6: 'dik-diks'}
        
        self.version = version
        if version == 'v1':
            self.metadata = pd.read_csv(file_path + "/metadata.csv", index_col=0)
        elif version == 'v2': 
            self.metadata = pd.read_csv(file_path + "/metadata_v2.csv", index_col=0)
        else: 
            raise Exception("Invalid dataset version.")
        
        self.metadata_for_split = self.metadata[self.metadata[shift] == split]

    def __len__(self): 
        return len(self.metadata_for_split)

    def __getitem__(self, index): 
        img_path = self.file_path + '/' + self.metadata_for_split.iloc[index]['img_path']
        class_label = self.metadata_for_split.iloc[index]['class_label'] 
        color = self.metadata_for_split.iloc[index]['color']
        time = self.metadata_for_split.iloc[index]['time']
        
        img = Image.open(img_path).convert("RGB") 
        
        if self.transform: 
            img = self.transform(img) 
        else: 
            img = img.resize((512, 512))

        return img, class_label, color, time


_DESCRIPTION = "Stable Diffusion: iWildCam source domain dataset."

class iWildCam_Dataset_Base(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
            datasets.BuilderConfig(name='color_day-to-night', version=VERSION), 
            datasets.BuilderConfig(name='grayscale_day-to-night', version=VERSION), 
            datasets.BuilderConfig(name='color_night-to-day', version=VERSION),
            datasets.BuilderConfig(name='grayscale_night-to-day', version=VERSION), 
            datasets.BuilderConfig(name='color-to-grayscale', version=VERSION), 
            datasets.BuilderConfig(name='grayscale-to-color', version=VERSION), 
        ]
    DEFAULT_CONFIG_NAME = "color_day-to-night"
    
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
                                      "root": "/mnt/scratch-lids/scratch/qixuanj/iwildcam_subset_organized",
                                      "split" : "train",
                                  }),
        ]
        
    def _generate_examples(self, root, split):
        shift = self.config.name
        # train_ds = iWildCamDataset(root, split="train", shift=shift)
        # val_ds = iWildCamDataset(root, split="val", shift=shift)
        # extra_ds = iWildCamDataset(root, split="extra", shift=shift)
        train_ds = iWildCamDataset(root, split="train", shift=shift, version='v2')
        val_ds = iWildCamDataset(root, split="val", shift=shift, version='v2')
        extra_ds = iWildCamDataset(root, split="extra", shift=shift, version='v2')
        mapping = train_ds.label_to_class_name
        
        split_data = torch.utils.data.ConcatDataset([train_ds, val_ds, extra_ds])
                                                
        for idx in range(len(split_data)): 
            item = split_data[idx]
            img, y, color, time = item[0], item[1], item[2], item[3]
            if shift == 'color_day-to-night' or shift == 'grayscale_day-to-night': 
                if time == 'night': 
                    setting = 'target-domain'
                else: 
                    setting = 'source-domain' 
            elif shift == 'color_night-to-day' or shift == 'grayscale_night-to-day': 
                if time == 'night': 
                    setting = 'source-domain'
                else: 
                    setting = 'target-domain'
            elif shift == 'color-to-grayscale': 
                if color == 'color': 
                    setting = 'source-domain' 
                else: 
                    setting = 'target-domain' 
            elif shift == 'grayscale-to-color': 
                if color == 'color': 
                    setting = 'target-domain' 
                else: 
                    setting = 'source-domain' 
                    
            
            prompt = f"a camera trap photo of {mapping[y]} with {setting}"
            
            yield idx, {
                "img": img,
                "prompt": prompt,
            }