"""
Provides access to the different metrics functions.
Each metric must take as input a tensor of targets and a tensor of predictions i.e. shape (N, C, H, W), respectively the number of samples, predicted likelihood per class, height, width.
"""

from sota_model.metrics.TorchMetrics import get_iou, get_dice

##
## The following dict can be modified to add metrics. 
##
metrics = {
    "iou": get_iou(),
    "dice": get_dice()
}

def load_metrics(alias, list_available=False):
    """
    Loads the metrics to measure the predictions.
    If `list_available` attribute is set to True, the function only prints all available aliases.
        
        metric = load_metrics("dice")
        score = metric(targets, preds)
    
    Note: both `targets` and `preds` are meant to be of shape (N, C, H, W) respectively the number of samples, predicted likelihood per class, height, width.
    
    Parameters:
        alias: str
            Alias for the metric to load.
        list_available: bool (default=False)
            Lists all available aliases.
    """
    
    if list_available:
        aliases = '  ' + '\n  '.join(metrics.keys())
        print(f"Available tasks aliases:\n{aliases}")
        return
        
    if not isinstance(alias, str):
        raise AttributeError('The task alias should be a string.')
        
    if not alias in metrics.keys():
        raise AttributeError('The metric alias is invalid. Please use load_metrics("", list_available=True) to see all available aliases.')

    metric = metrics[alias]
    
    return metric