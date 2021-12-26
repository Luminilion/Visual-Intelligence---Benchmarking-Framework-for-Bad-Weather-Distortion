"""
Performs a semantic segmentation task with 19 classes taken from the Cityscapes dataset.
This method creates a function given a model. The function can then be used on data to perform semantic segmentation. 
"""

import torch.nn as nn
import torch
import numpy as np

colors_data = [ [0,  0,  0],
          [128, 64,128],
          [244, 35,232],
          [70, 70, 70],
          [102,102,156],
          [190,153,153],
          [153,153,153],
          [250,170, 30],
          [250,170, 30],
          [107,142, 35],
          [152,251,152],
          [ 70,130,180],
          [220, 20, 60],
          [255,  0,  0],
          [ 0,  0,142],
          [  0,  0, 70],
          [  0,  0, 70],
          [  0, 80,100],
          [  0,  0,230],
          [119, 11, 32] ]

def convert_to_RGB(y):
    """
    Identifies the class by taking the argmax across values.
    Replaces the class id by the corresponding color.
    
    Returns 
        the identified classes `idtfd` with each pixel assigned with a class i.e. shape of (N, C, H, W) with C the class id,
        the colored images `imgs` with each pixel assigned a color corresponding to the class i.e. shape (N, 3, H, W)
    """
    # n_classes = 19
    n_classes = y.shape[1]
    
    # Create color attribution for coloring
    colors = np.array(colors_data[:n_classes])
    print(f"Using {len(colors)} classes.")
    
    # Identify the class
    idtfd = y.argmax(dim=1).numpy()

    # Create the colored images
    imgs = colors[idtfd]
    imgs = imgs.transpose(0, 3, 1, 2)
    
    return idtfd, imgs

def get_semantic_RGB(model):
    """
    Creates a semantic segmentation task for a single model. 
    Returns a fonction that can be used on a list of RGB (3 channels) images.
    
    Note: the code makes use of the model as an implementation of the `torch.nn.Module`.
    
        semantic_task = get_semantic(UNet)
        preds = semantic_task(imgs)
        preds, colored_preds = semantic_task(imgs, get_colored=True)
        preds, classified_preds = semantic_task(imgs, get_classification=True)

    Parameters:
        model: torch.Tensor
            The model to use for the segmentation
    """
    
    if not isinstance(model, nn.Module):
        raise AttributeError("The passed model should be an implementation of the `torch.nn.Module` class.")
    
    def semantic_segmentation_RGB(imgs, get_colored=False, get_classification=False):
        """
        Performs the semantic segmentation task with the input model.
        Note: the code makes use of the model as an implementation of the `torch.nn.Module`.
    
            semantic_task = get_semantic(UNet)
            preds = semantic_task(imgs)
            preds, colored_preds = semantic_task(imgs, get_colored=True)
            preds, classified_preds = semantic_task(imgs, get_classification=True)
            
        Parameters: 
            imgs: torch.Tensor
                The list of RGB images to segment.
            get_colored: bool (default=False)
                Parameter indicating wether to return the colored images as well. One color per class.
            get_classification: bool (default=False)
                Parameter indicating wether to return the classified pixels. Most confident class per pixel.
                
        """
        if not isinstance(imgs, torch.Tensor):
            raise AttributeError("The input for the segmentation task should be a tensor of images.")
        if not imgs.shape[1] == 3 or len(imgs.shape)!=4:
            raise AttributeError("The input should have the following dimensons (N, C, H, W), respectively number of images, channels, height and width.")
        
        # Setting the model in eval mode
        model.eval()
                
        # running the images through the model
        results = model(imgs)
        
        # Identifying class and coloring.
        idtfd, colored = convert_to_RGB(results)
        
        output = [results]
        
        if get_colored:
            output.append(colored)
        if get_classification:
            output.append(idtfd)
        return output
        
    return semantic_segmentation_RGB