"""
Provides a Cityscapes dataloader.

IMPORTANT: the file requires that the data is provided at the `data` location. Default location being `ROOT\data\cityscapes`.
The data should be in a standard way (folder structure as provided by the dataset's source). For custom datasets, it can be handled as chosen by the author of the custom dataloader.

Reference: https://www.cityscapes-dataset.com/
"""

import torchvision.datasets as D

def get_cityscapes(location='../../data/cityscapes'):
    """
    Provides the standard cityscapes dataset.
    This class uses the Pytorch dataloader created for the dataset.
    """
    try:
        dataset = D.Cityscapes(location, split="train", target_type='semantic')
    except ValueError:
        raise RuntimeError("There was an error loading your data. Try using the `location` attribute of `load_data` to indicate your data's folder location.")
    
    return dataset