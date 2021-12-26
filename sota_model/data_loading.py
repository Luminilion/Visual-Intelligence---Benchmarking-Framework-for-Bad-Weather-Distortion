"""
Provides the various dataloaders. The data is loaded from the given sources. 
If one wants to add its dataset here, the dictionary must be updated.

IMPORTANT: the datasets should be loaded in evaluation mode if any specification is necessary.
IMPORTANT: the datasets should return `PIL.Image` type data samples.

The loader's creation method should support the following arguments.
- location (str): an optional string specifying the dataset's location. 
"""

from sota_model.dataloaders.Cityscapes import get_cityscapes

##
## The following dict can be modified to add dataloaders. 
##
dataloaders = {
    "cityscapes": get_cityscapes,
    # "cityscapes_bad_weather": get_cs_augmented_loader,
}

def load_data(alias, list_available=False, location=None):
    """
    Loads a dataloader providing access to the data to be used for the semantic segmentation evaluation.
    If `list_available` attribute is set to True, the function only prints all available aliases.
        
        data = load_data("Cityscapes")
        img = data[0][0]
        gt = data[0][1]
    
    Note: `img` and `data` are expected to be of type `PIL.Image`
    
    Parameters:
        alias: str
            Alias for the dataset to load.
        list_available: bool (default=False)
            Lists all available aliases.
    """
    
    if list_available:
        aliases = '  ' + '\n  '.join(dataloaders.keys())
        print(f"Available dataloaders aliases:\n{aliases}")
        return
        
    if not isinstance(alias, str):
        raise AttributeError('The dataloader alias should be a string.')
        
    if not alias in dataloaders.keys():
        raise AttributeError('The dataloader alias is invalid. Please use load_data("", list_available=True) to see all available aliases.')
    
    if location is not None:
        loader = dataloaders[alias](location=location)
    else:
        loader = dataloaders[alias]()
    
    return loader