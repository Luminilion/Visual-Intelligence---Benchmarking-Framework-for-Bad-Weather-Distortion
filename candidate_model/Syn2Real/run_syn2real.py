import numpy as np
import torch
import random
from model import DeRain_v2
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import argparse
import os
from os import listdir
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.utils as utils


class DataIterator(data.Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.image_names = listdir(self.dataset_dir)

    def _format_image(self, index):
        name = self.image_names[index]
        img = Image.open(os.path.join(self.dataset_dir, name))

        # Resizing image in the multiple of 16"
        wd_new, ht_new = img.size
        if ht_new > wd_new and ht_new > 1024:
            wd_new = int(np.ceil(wd_new * 1024 / ht_new))
            ht_new = 1024
        elif ht_new <= wd_new and wd_new > 1024:
            ht_new = int(np.ceil(ht_new * 1024 / wd_new))
            wd_new = 1024
        wd_new = int(16 * np.ceil(wd_new / 16.0))
        ht_new = int(16 * np.ceil(ht_new / 16.0))
        img = img.resize((wd_new, ht_new), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img = transform_input(img)

        return img, name

    def __getitem__(self, index):
        res = self._format_image(index)
        return res

    def __len__(self):
        return len(self.image_names)


class RunSyn2Real:

    def __init__(self, args):
        # Assign arguments passed
        self.dataset_dir = args.d
        self.output_dir = args.o


    def _save_image(self, pred_image, image_name, dir):
        pred_image_images = torch.split(pred_image, 1, dim=0)
        batch_num = len(pred_image_images)

        for ind in range(batch_num):
            image_name_1 = image_name[ind].split('/')[-1]
            utils.save_image(pred_image_images[ind], os.path.join(dir, image_name_1[:-3] + 'png'))


    def predict_with_model(self):
        # Generate derained images with model
        ## Set variables
        batch_size=1
        exp_name = "DDN_SIRR_withGP"
        category = "derain"
        seed = 19

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)

        ## GPU
        device_ids = [Id for Id in range(torch.cuda.device_count())]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ## Load model
        net = DeRain_v2()
        net = net.to(device)
        net = nn.DataParallel(net, device_ids=device_ids)
        net.load_state_dict(torch.load('../candidate_model/Syn2Real/{}/{}_best'.format(exp_name, category), map_location=device))

        ## Pass images
        loader = DataLoader(DataIterator(self.dataset_dir), batch_size=batch_size, shuffle=False, num_workers=8)
        for id, data in enumerate(loader):
            with torch.no_grad():
                img, name = data
                img = img.to(device)
                prediction, _ = net(img)  # Predict with model
            self._save_image(prediction, name, self.output_dir)  # Save predicted image in output directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for running the model with the benchmarking framework.')
    parser.add_argument("-d", help="Dataset directory", type=str)
    parser.add_argument("-o", help="Output directory", type=str)
    args = parser.parse_args()

    runsyn2real = RunSyn2Real(args)
    runsyn2real.predict_with_model()