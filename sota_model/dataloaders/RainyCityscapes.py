"""
DataLoader for the Weather Cityscapes.
It handles the rainy Cityscapes as explained here : https://team.inria.fr/rits/computer-vision/weather-augment/
"""

from torch.utils.data import Dataset
import torchvision.datasets as D
from PIL import Image

import os
import pandas as pd

class RainyCityscapes(Dataset):
    def __init__(self, img_dir, cityscapes_dir, rain_diff='5mm'):
        self.img_labels = D.Cityscapes(cityscapes_dir, split="val", target_type='semantic')
        self.img_dir = img_dir + '/leftImg8bit/train/rain_diff/'+rain_diff+'/rainy_image'
    
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

