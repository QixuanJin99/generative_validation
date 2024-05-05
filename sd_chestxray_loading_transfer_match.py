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


_DESCRIPTION = "Stable Diffusion: CXR transfer domain dataset."
            
class CXR_Dataset_Base(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
            datasets.BuilderConfig(name="mimic_10_match", version=VERSION), 
            datasets.BuilderConfig(name="mimic_50_match", version=VERSION), 
            datasets.BuilderConfig(name="mimic_100_match", version=VERSION),
            datasets.BuilderConfig(name="mimic_250_match", version=VERSION), 
            datasets.BuilderConfig(name="mimic_500_match", version=VERSION), 
            datasets.BuilderConfig(name="mimic_1000_match", version=VERSION), 
            datasets.BuilderConfig(name="chexpert_10_match", version=VERSION), 
            datasets.BuilderConfig(name="chexpert_50_match", version=VERSION), 
            datasets.BuilderConfig(name="chexpert_100_match", version=VERSION),
            datasets.BuilderConfig(name="chexpert_250_match", version=VERSION), 
            datasets.BuilderConfig(name="chexpert_500_match", version=VERSION), 
            datasets.BuilderConfig(name="chexpert_1000_match", version=VERSION),
            datasets.BuilderConfig(name="padchest_10_match", version=VERSION), 
            datasets.BuilderConfig(name="padchest_50_match", version=VERSION), 
            datasets.BuilderConfig(name="padchest_100_match", version=VERSION),
            datasets.BuilderConfig(name="padchest_250_match", version=VERSION), 
            datasets.BuilderConfig(name="padchest_500_match", version=VERSION), 
            datasets.BuilderConfig(name="padchest_1000_match", version=VERSION),
            datasets.BuilderConfig(name="nih_10_match", version=VERSION), 
            datasets.BuilderConfig(name="nih_50_match", version=VERSION), 
            datasets.BuilderConfig(name="nih_100_match", version=VERSION),
            datasets.BuilderConfig(name="nih_250_match", version=VERSION), 
            datasets.BuilderConfig(name="nih_500_match", version=VERSION), 
            datasets.BuilderConfig(name="nih_1000_match", version=VERSION),
        ]
    DEFAULT_CONFIG_NAME = "mimic_1000_match"
    
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
                                      "file_path": "/data/healthy-ml/scratch/qixuanj/generative_validation/cxr_prompt_files_transfer_seed0.pkl",
                                  }),
        ]
        
    def _generate_examples(self, split, file_path):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        img_dirs = {"mimic": "/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG/files", 
                    "padchest": "/data/healthy-ml/gobi1/data/PadChest/images-224", 
                    "chexpert": "/data/healthy-ml/gobi1/data", 
                    "nih": "/data/healthy-ml/gobi1/data/ChestXray8/images"}

        dataset_name = self.config.name.split("_")[0]
        train_num = int(self.config.name.split("_")[1])
        config = self.config.name.split("_")[2]

        split_data = self.data[dataset_name][train_num][f"{config}_ds"].reset_index(drop=True)

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