import pandas as pd
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision import transforms
import pickle
from skimage.io import imread
import torchxrayvision as xrv


class CXRGenDataset(VisionDataset):
    def __init__(self, file_path, transfer_dataset, split='train', transform=None):
        super(CXRGenDataset, self).__init__(root=None, transform=transform)
        self.file_path = file_path
        self.transfer_dataset = transfer_dataset 
        self.split = split
        self.transform = transform
        self.pathologies = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
                           "Lesion", "Pneumonia", "Pneumothorax", "No Finding"]

        with open("/data/healthy-ml/scratch/qixuanj/generative_validation/cxr_prompt_files_base10000.pkl", "rb") as f: 
            metadata = pickle.load(f)
        self.metadata = metadata = metadata['mimic']['train']

        if split == "gen_0.9": 
            self.metadata_for_split = self.metadata.dropna(subset=f'{self.transfer_dataset}_gen_0.9')
            self.img_path_col = f"{self.transfer_dataset}_gen_0.9"
        else: 
            raise Exception("Version of generated iamges dataset not implemented") 
            

    def __len__(self): 
        return len(self.metadata_for_split)

    def __getitem__(self, index): 
        if self.split == "gen_0.7" or self.split == "gen_0.9": 
            img_path = self.metadata_for_split.iloc[index][self.img_path_col]
        else: 
            img_path = self.file_path + '/' + self.metadata_for_split.iloc[index][self.img_path_col]
            
        label_names = self.metadata_for_split.iloc[index]['labels'] 
        labels = np.zeros(len(self.pathologies))
        label_indices = [i for i in range(len(self.pathologies)) if self.pathologies[i] in label_names]
        labels[label_indices] = 1

        # Same preprocessing as MIMIC-CXR images
        img = imread(img_path)
        img = xrv.utils.normalize(img, maxval=255, reshape=True)
        
        if self.transform: 
            img = self.transform(img) 
        else: 
            img = img.resize((224, 224))

        return {'idx': index, 'img': img, 'lab': labels}