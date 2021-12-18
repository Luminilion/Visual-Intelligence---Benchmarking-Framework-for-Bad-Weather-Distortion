import numpy as np
import torch
import random
from model import DeRain_v2
import torch.nn as nn
import argparse
import os
from os import listdir
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.utils as utils


class RunSyn2Real:

    def __init__(self):
        # Check and assign arguments passed
        parser = argparse.ArgumentParser(description='Parameters for running the model with the benchmarking framework.')
        parser.add_argument("-d", help="Dataset directory", type=str)
        parser.add_argument("-o", help="Output directory", type=str)
        args = parser.parse_args()

        self.dataset_dir = args.d
        self.output_dir = args.o


    def _format_image(self, dir, name):
        img = Image.open(os.path.join(dir, name))

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

        return img


    def _save_image(self, pred_image, image_name, dir):
        pred_image_images = torch.split(pred_image, 1, dim=0)
        batch_num = len(pred_image_images)

        for ind in range(batch_num):
            image_name_1 = image_name[ind].split('/')[-1]
            utils.save_image(pred_image_images[ind], os.path.join(dir, image_name_1[:-3] + 'png'))


    def predict_with_model(self):
        # Generate derained images with model
        ## Set variables
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
        net.load_state_dict(torch.load('./{}/{}_best'.format(exp_name, category), map_location=device))

        ## Pass images
        image_names = listdir(self.dataset_dir)
        for image in image_names:
            img = self._format_image(self.dataset_dir, image)
            prediction, _ = net(img)  # Predict with model
            self._save_image(prediction, image, self.output_dir)  # Save predicted image in output directory
