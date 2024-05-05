import datasets
import json
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
import pandas as pd
from glob import glob
import os
import pickle

import datasets


_DESCRIPTION = "Stable Diffusion: CXR source domain dataset."
            
class CXR_Dataset_Base(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")
    
    BUILDER_CONFIGS = [
            datasets.BuilderConfig(name="mimic_base", version=VERSION, description="MIMIC base dataset with 10000 samples"), 
            datasets.BuilderConfig(name="chexpert_base", version=VERSION, description="CheXpert base dataset with 10000 samples"),
            datasets.BuilderConfig(name="padchest_base", version=VERSION, description="PadChest base dataset with 10000 samples"),
            datasets.BuilderConfig(name="nih_base", version=VERSION, description="NIH base dataset with 10000 samples"),
        ]
    DEFAULT_CONFIG_NAME = "mimic_base"
    
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
                                      "file_path": "/data/healthy-ml/scratch/qixuanj/generative_validation/cxr_prompt_files_base10000.pkl",
                                  }),
            datasets.SplitGenerator(name=datasets.Split.TEST, 
                                  gen_kwargs={
                                      "split" : "test",
                                      "file_path": "/data/healthy-ml/scratch/qixuanj/generative_validation/cxr_prompt_files_base10000.pkl",
                                  })
        ]
        
    def _generate_examples(self, split, file_path):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        img_dirs = {"mimic": "/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG/files", 
                    "padchest": "/data/healthy-ml/gobi1/data/PadChest/images-224", 
                    "chexpert": "/data/healthy-ml/gobi1/data", 
                    "nih": "/data/healthy-ml/gobi1/data/ChestXray8/images"}

        if self.config.name == "mimic_base":
            split_data = self.data["mimic"][split].reset_index(drop=True)
        elif self.config.name == "chexpert_base":
            split_data = self.data["chexpert"][split].reset_index(drop=True)
        elif self.config.name == "padchest_base":
            split_data = self.data["padchest"][split].reset_index(drop=True)
        elif self.config.name == "nih_base":
            split_data = self.data["nih"][split].reset_index(drop=True)
        
        for idx in range(len(split_data)): 
            item = split_data.iloc[idx]

            filename = item['file_suffix']
            img_dir = img_dirs[item['dataset_name']]
            prompt = f"a radiograph from dataset {item['dataset_name']} with conditions {item['labels']}"

            target = cv2.imread(f"{img_dir}/{filename}")
            target = cv2.resize(np.array(target), (512, 512))
            
            yield filename, {
                "img": target,
                "prompt": prompt,
            }