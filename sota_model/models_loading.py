"""
Helper file to create all models.

Models can be loaded with the `load_model()` function. 
They have to be registered under an alias in the `models` dict.
They have to implement a class in the `models/` folder.

"""
from models.SINet import get_sinet
from models.UNet import get_unet

##
## The following dict can be modified to add models. 
##
models = {
    "SINet": get_sinet(),
    "UNet": get_unet(),
    "HRNet":""
}

def load_model(alias, list_available=False):
    """
    Loads a model to be used as an implementation of the torch.nn.Module class.
    If `list_available` attribute is set to True, the function only prints all available aliases.
        
        model = load_model("SINet")
        preds = model(x)
        
    Parameters:
        alias: str
            Alias for the model to load.
        list_available: bool (default=False)
            Lists all available aliases.
    """
    if list_available:
        aliases = '  ' + '\n  '.join(models.keys())
        print(f"Available models aliases:\n{aliases}")
        return
        
    if not isinstance(alias, str):
        raise AttributeError('The model alias should be a string.')
        
    if not alias in models.keys():
        raise AttributeError('The model alias is invalid. Please use load_model("", list_available=True) to see all available aliases.')
    
    model = models[alias]
    
    return model