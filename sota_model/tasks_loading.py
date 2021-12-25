"""
Provides the tasks done to evaluate the candidate model.
The tasks should take as input a list of images.
"""


##
## The following dict can be modified to add tasks. 
##
tasks = {
    "semantic": semantic,
}

def load_tasks(alias, list_available=False, location=None):
    """
    Loads the tasks to execute on the data in order to evaluate the candidate model.
    If `list_available` attribute is set to True, the function only prints all available aliases.
        
        task = load_tasks("semantic")
        t_imgs = task(imgs)
        t_denoized = task(denoized)
        t_augmented = task(augmented)
        t_augdenoized = task(augdenoized)
    
    Note: `imgs` and corresponding are expected to be lists of `PIL.Image`
    
    Parameters:
        alias: str
            Alias for the model to load.
        list_available: bool (default=False)
            Lists all available aliases.
    """
    
    if list_available:
        aliases = '  ' + '\n  '.join(dataloaders.keys())
        print(f"Available dataloaders aliases:\n{aliases}")
        return
        
    if not isinstance(alias, str):
        raise AttributeError('The task alias should be a string.')
        
    if not alias in dataloaders.keys():
        raise AttributeError('The dataloader alias is invalid. Please use load_data("", list_available=True) to see all available aliases.')
    
    if location is not None:
        loader = dataloaders[alias](location=location)
    else:
        loader = dataloaders[alias]()
    
    return loader