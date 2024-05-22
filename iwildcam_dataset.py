import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision import transforms

class iWildCamDataset(VisionDataset):
    def __init__(self, file_path, shift, split='train', transform=None, version='v3', filter=None, branches=None):
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
        self.filter = filter
        self.branches = branches
        
        if version == 'v1':
            self.metadata = pd.read_csv(file_path + "/metadata.csv", index_col=0)
        elif version == 'v2': 
            self.metadata = pd.read_csv(file_path + "/metadata_v2.csv", index_col=0)
        elif version == 'v3': 
            self.metadata = pd.read_csv(file_path + "/metadata_v3.csv", index_col=0)
        elif version == 'v4': 
            self.metadata = pd.read_csv(file_path + "/metadata_v4.csv", index_col=0)
        elif version == 'copy': 
            self.metadata = pd.read_csv(file_path + "/metadata_copy.csv", index_col=0)
        elif version == 'copy2': 
            self.metadata = pd.read_csv(file_path + "/metadata_copy2.csv", index_col=0)
        else: 
            raise Exception("Invalid dataset version.")
        

        if split == "gen_0.7": 
            if self.filter: 
                self.img_path_col = f"{self.shift}_generate_0.7_{self.filter}"
            else:
                self.img_path_col = f"{self.shift}_generate_0.7"
            self.metadata_for_split = self.metadata[self.metadata[self.img_path_col].notnull()]
        elif split == "gen_0.9": 
            if self.filter: 
                self.img_path_col = f"{self.shift}_generate_0.9_{self.filter}"
            else:
                self.img_path_col = f"{self.shift}_generate_0.9"
            self.metadata_for_split = self.metadata[self.metadata[self.img_path_col].notnull()]
        elif split == "gen_1.0": 
            if self.filter: 
                self.img_path_col = f"{self.shift}_generate_1.0_{self.filter}"
            else:
                self.img_path_col = f"{self.shift}_generate_1.0"
            self.metadata_for_split = self.metadata[self.metadata[self.img_path_col].notnull()]
        elif "branches" in split: 
            self.img_path_col = split 
            self.metadata_for_split = self.metadata[self.metadata[self.img_path_col].notnull()]
        elif branches: 
            self.metdata_for_split = self.metadata[self.metatdata[f"{self.shift}_branch"] == branches]
            self.img_path_col = 'img_path'
        else: 
            self.metadata_for_split = self.metadata[self.metadata[shift] == split]
            self.img_path_col = 'img_path'
            
            

    def __len__(self): 
        return len(self.metadata_for_split)

    def __getitem__(self, index): 
        if self.split == "gen_0.7" or self.split == "gen_0.9" or self.split == "gen_1.0" or "branches" in self.split: 
            img_path = self.metadata_for_split.iloc[index][self.img_path_col]
        else: 
            img_path = self.file_path + '/' + self.metadata_for_split.iloc[index][self.img_path_col]
        class_label = self.metadata_for_split.iloc[index]['class_label'] 
        color = self.metadata_for_split.iloc[index]['color']
        time = self.metadata_for_split.iloc[index]['time']
        
        img = Image.open(img_path).convert("RGB") 
        
        if self.transform: 
            img = self.transform(img) 
        else: 
            img = img.resize((512, 512))

        return img, class_label, color, time