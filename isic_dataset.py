import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision import transforms

class SpuriousDermDataset(VisionDataset):
    def __init__(self, file_path, split='train', transform=None, get_mask=False, gen_version=None):
        super(SpuriousDermDataset, self).__init__(root=None, transform=transform)

        self.file_path = file_path
        self.split = split
        self.transform = transform
        self.label_map = {'malignant': 1, 'benign': 0}
        self.get_mask = get_mask
        self.gen_version = gen_version

        if self.gen_version: 
            self.metadata = pd.read_csv(file_path+'metadata_gen.csv', index_col=0)
            self.metadata_for_split = self.metadata[self.metadata[self.gen_version].notnull()]
        else: 
            # Load metadata from CSV
            self.metadata = pd.read_csv(file_path+'metadata.csv')
            # Filter metadata based on split
            self.metadata_for_split = self.metadata.iloc[[self.split in x for x in self.metadata['image']]].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata_for_split)

    def __getitem__(self, index):
        if self.get_mask: 
            img_path = self.file_path + self.metadata_for_split.iloc[index]['mask']
            img = Image.open(img_path).convert('L')
        elif self.gen_version: 
            img_path = self.metadata_for_split.iloc[index][self.gen_version]
            img = Image.open(img_path).convert('RGB')
        else: 
            img_path = self.file_path + self.metadata_for_split.iloc[index]['image']
            img = Image.open(img_path).convert('RGB')
        melanoma_label = self.label_map[self.metadata_for_split.iloc[index]['benign_malignant']]
        group_label = self.metadata_for_split.iloc[index]['class']

        if self.transform:
            img = self.transform(img)
        else: 
            img = img.resize((512, 512))

        return img, melanoma_label, group_label