"""
DataLoader for the Weather Cityscapes.
It handles the foggy Cityscapes as explained here : https://github.com/sakaridis/fog_simulation-SFSU_synthetic/
"""

from torch.utils.data import Dataset
import torchvision.datasets as D
from PIL import Image

import os

class FoggyCityscapes(Dataset):
    def __init__(self, img_dir, cityscapes_dir, transmittance='30m'):
        self.img_labels = D.Cityscapes(cityscapes_dir, split="train", target_type='semantic')
        self.img_dir = img_dir + '/leftImg8bit/train/fog_transmittance/' + transmittance
    
        cities = os.listdir(self.img_dir)
        self.img_per_folder = {}
        for f in cities:
            imgs = os.listdir(self.img_dir +'/' + f)
            self.img_per_folder[f] = imgs

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        iterator, memory, city_id = 0, 0, 0
        while iterator < idx:
            memory = iterator
            c = list(self.img_per_folder.keys())[city_id]
            iterator += len(self.img_per_folder[c])
            city_id+=1
        
        city = list(self.img_per_folder.keys())[city_id]
        idx = idx-memory
        
        print(self.img_dir + '/' + self.img_per_folder[city][idx])
        image = Image.open(self.img_dir + '/' + city + '/' + self.img_per_folder[city][idx])
        label = self.img_labels[idx][1]
        
        return image, label

