"""
Loads a pretrained UNet.
"""

import segmentation_models_pytorch as smp
import torchvision.datasets as D
import matplotlib.pyplot as plt
import torch
import numpy as np


def get_unet(in_channels = 3, n_classes=19, encoder='resnet34'):
    """
    Gets a pretrained UNet from the segmentation_models_pytorch module.
    Source here: https://smp.readthedocs.io/en/latest/
    """
    
    model = smp.Unet(
        encoder_name=encoder,        # choose encoder, refer to documentation
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=n_classes,                      # model output channels (number of classes in your dataset)
    )
    
    return model
    
    