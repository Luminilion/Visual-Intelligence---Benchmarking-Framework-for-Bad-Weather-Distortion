"""
Provides the tasks done to evaluate the candidate model.
Each task should take as input a list of images and a model.
"""

from sota_model.tasks.SemanticSegmentation import get_semantic_RGB


##
## The following dict can be modified to add tasks. 
##
tasks = {
    "semantic": get_semantic_RGB,
}

def load_tasks(alias, list_available=False):
    """
    Loads the tasks to execute on the data in order to evaluate the candidate model.
    If `list_available` attribute is set to True, the function only prints all available aliases.
        
        task = load_tasks("semantic")
        task_unet = task(UNet)
        
        t_imgs = task_unet(imgs)
        t_denoized = task_unet(denoized)
        t_augmented = task_unet(augmented)
        t_augdenoized = task_unet(augdenoized)
    
    Note: `imgs` and corresponding are expected to be lists of `PIL.Image`
    
    Parameters:
        alias: str
            Alias for the model to load.
        list_available: bool (default=False)
            Lists all available aliases.
    """
    
    if list_available:
        aliases = '  ' + '\n  '.join(tasks.keys())
        print(f"Available tasks aliases:\n{aliases}")
        return
        
    if not isinstance(alias, str):
        raise AttributeError('The task alias should be a string.')
        
    if not alias in tasks.keys():
        raise AttributeError('The task alias is invalid. Please use load_tasks("", list_available=True) to see all available aliases.')

    task = tasks[alias]
    
    return task