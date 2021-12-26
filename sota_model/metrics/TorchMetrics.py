"""
Metrics access methods for TorchMetrics metrics.
Contains the following : 
    - IoU
    - Dice
"""

import torchmetrics.functional as tmf

def get_iou():
    """
    Returns a function computing the jaccard index over 2 inputs.
    Inputs are predictions and targets, i.e. tensors of size (N, C, H, W), respectively the number of samples, predicted likelihood per class, height, width.
    
    Parameters:
        predictions: torch.Tensor
            The predictions from semantic segmentation.
        targets: torch.Tensor
            The targets from ground truth.
    """
    
    def iou(preds, targets):
        """
        Computes the IoU score of the predictions with respect to the target.
        Note: targets should be a tensor of integers.
        
        Parameters:
            preds: torch.Tensor
                Tensor of predictions. Should be of shape (N, C, H, W), respectively the number of samples, predicted likelihood per class, height, width.
            targets: torch.Tensor
                Tensor of target classes. Should be of shape (N, C, H, W), respectively the number of samples, predicted likelihood per class, height, width.
        """
        return tmf.iou(preds, targets)
    
    return iou

def get_dice():
    """
    Returns a function computing the dice score over 2 inputs.
    Inputs are predictions and targets, i.e. tensors of size (N, C, H, W), respectively the number of samples, predicted likelihood per class, height, width.
    
    Parameters:
        predictions: torch.Tensor
            The predictions from semantic segmentation.
        targets: torch.Tensor
            The targets from ground truth.
    """
    
    def dice_score(preds, targets):
        """
        Computes the dice score of the predictions with respect to the target.
        
        Parameters:
            preds: torch.Tensor
                Tensor of predictions. Should be of shape (N, C, H, W), respectively the number of samples, predicted likelihood per class, height, width.
            targets: torch.Tensor
                Tensor of target classes. Should be of shape (N, C, H, W), respectively the number of samples, predicted likelihood per class, height, width.
        """
        
        return tmf.dice_score(preds, targets)
    
    return dice_score
