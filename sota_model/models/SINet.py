"""
Returns a SINet version pretrained on Imagenet.
"""

from pytorchcv.model_provider import get_model
import torch

def get_sinet() :
    """
    Gets a pretrained version of the SINet. The version is pretrained on ImageNet.
    This method uses the PytorchCV framework available here: https://pypi.org/project/pytorchcv/
    """
    net = get_model("sinet_cityscapes", pretrained=True)
    return net