"""
Utils files for functions not directly playing a role in the classification pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from torch import Tensor

def store_results(imgs, path, filename="result_"):
    """
    Stores the result of a task in a specific folder.
    Note: The input to be saved should come in the shape (N, C, H, W), respectively number of samples, channels, height and width
    
    Parameters:
        imgs: list-like convertible to np.ndarray (torch.tensor, np.ndarray)
            List of results to save.
        path: str
            Directory path in which to store the results. If it does not exist, it will be created.
        filename: str (default='result_')
            Filename prefix appended in front of the file id.
    """
    
    if isinstance(imgs, Tensor):
        imgs_a = imgs.detach().numpy()
        imgs_a = np.array(imgs_a, dtype=np.uint8)
    else:
        imgs_a = np.array(imgs, dtype=np.uint8)
    # Image needs to be saved as either (MxN) or (MxNx3) or (MxNx4)
    imgs_a = imgs_a.transpose(0, 2, 3, 1)
    print(imgs_a.shape)
    if imgs_a.shape[0] == 1:
        restore_first_axis = True
    imgs_a = imgs_a.squeeze()
    if restore_first_axis:
        imgs_a = imgs_a[np.newaxis, :]
    print(imgs_a.shape)
    
    if not os.path.isdir(path):
        print(f"No folder found at {path}. Creating one to save results.")
        os.mkdir(path)
    
    for i, f in enumerate(imgs_a):
        name = path + filename +  str(i) + '.png'
        print(name)
        plt.imsave(name, f)
    
