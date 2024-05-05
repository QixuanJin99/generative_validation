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


_DESCRIPTION = "Stable Diffusion fine-tuning dataset for cross-domain chest X-rays "

'''
NOTE: for some reason, multiple dataset configs dont' work 
'''
class CXRDatasetConfig(datasets.BuilderConfig): 
    def __init__(self, **kwargs):
        super(CXRDatasetConfig, self).__init__(**kwargs)
        self.features = ['img', 'prompt']
        self.use_datasets = kwargs.pop("use_datasets", ["mimic", "chexpert", "nih", "padchest"])
            
class SD_ChestXray(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = CXRDatasetConfig
    BUILDER_CONFIGS = [
            CXRDatasetConfig(name = "sd_chestxray", 
            description = "Chest X-ray images [MIMIC-CXR, PadChest, CheXpert, ChestXray14], text prompts with domain, views, disease label", 
            )
        ]
    
    def _info(self): 
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
            {
                "img": datasets.Image(), 
                "prompt": datasets.Value("string"),
            }), 
            supervised_keys=None, 
        )
    
    def _split_generators(self, dl_manager): 
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, 
                                  gen_kwargs={
                                      "split" : "train",
                                      "img_dirs" : {"mimic": "/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG/files", 
                                                    "padchest": "/data/healthy-ml/gobi1/data/PadChest/images-224", 
                                                    "chexpert": "/data/healthy-ml/gobi1/data", 
                                                    "nih": "/data/healthy-ml/gobi1/data/ChestXray8/images"}
                                  }), 
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, 
                                  gen_kwargs={
                                      "split" : "val",
                                      "img_dirs" : {"mimic": "/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG/files", 
                                                    "padchest": "/data/healthy-ml/gobi1/data/PadChest/images-224", 
                                                    "chexpert": "/data/healthy-ml/gobi1/data", 
                                                    "nih": "/data/healthy-ml/gobi1/data/ChestXray8/images"}
                                  }),
            datasets.SplitGenerator(name=datasets.Split.TEST, 
                                  gen_kwargs={
                                      "split" : "test",
                                      "img_dirs" : {"mimic": "/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG/files", 
                                                    "padchest": "/data/healthy-ml/gobi1/data/PadChest/images-224", 
                                                    "chexpert": "/data/healthy-ml/gobi1/data", 
                                                    "nih": "/data/healthy-ml/gobi1/data/ChestXray8/images"}
                                  })
        ]
        
    def _generate_examples(self, split, img_dirs):
        with open("/data/healthy-ml/scratch/qixuanj/generative_validation/cxr_prompt_files.pkl", 'rb') as f:
            self.data = pickle.load(f)
            
        split_data = pd.DataFrame()
        for dataset_name in img_dirs.keys(): 
            tmp = self.data[dataset_name]
            split_data = pd.concat([split_data, tmp[tmp['split'] == split]])
        split_data = split_data.reset_index()
        
        for idx in range(len(split_data)): 
            item = split_data.iloc[idx]

            filename = item['file_suffix']
            img_dir = img_dirs[item['dataset_name']]
            prompt = f"a radiograph from dataset {item['dataset_name']} with conditions {item['labels']}"

            target = cv2.imread(f"{img_dir}/{filename}")
            target = cv2.resize(np.array(target), (512, 512))
            # target = np.array(target).astype(np.float32) / 255.0
            
            yield filename, {
                "img": target,
                "prompt": prompt,
            }