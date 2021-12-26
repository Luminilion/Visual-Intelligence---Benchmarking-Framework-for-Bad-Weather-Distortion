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
    
    
from models_loading import models
from tasks_loading import tasks
from metrics_loading import metrics

from collections import defaultdict
from datetime import datetime
    
class Report:
    """
    Report class used to produce the textual report from the benchmarking framework.
    """
    
    def __init__(self):
        self.title = "###### REPORT"
        
        self.datasets_title = "#### Datasets used"
        self.datasets = []
        
        self.tasks_title = '#### Tasks'
        self.task_models_title = "#### Tasks models used"
        
        self.tasks_models = {}
        
        self.time_title = "#### Time spent"
        self.time = "0:00:00"
        
        self.metrics_title = "#### Metrics Results"
        self.metrics_prefix = "##"
        self.metrics = defaultdict(dict)
        
    def add_dataset(self, dt):
        """
        Adds a dataset information to the report.
        
        Parameters:
            dt: str
                Name of the dataset.
        """
        self.datasets.append(dt)
        
    def add_task_model(self, model, task):
        """
        Adds a model for a task to the report.
        
        Parameters:
            model: str
                Name of the model to add.
            task: str
                Name of the task to add.
        """
        
        if not model in models.keys():
            raise AttributeError("The input model should be part of the usable models list. Verify the models_loading.py file.")
        if not task in tasks.keys():
            raise AttributeError("The input task should be part of the usable tasks list. Verify the tasks_loading.py file.")
            
        self.tasks_models[model]= task
        
    def add_metric(self, metric, score, imgs_type):
        """
        Adds a metric score to the report.
        The metric score is composed of the metric, the score according to this metric and the type of data evaluated with the metric.
        The type should be one of ["RAW", "DENOISED", "RAW_NOISY", "NOISY_DENOISED"].
        
        Parameters:
            metric: str
                The name of the metric used.
            score: str
                The score obtained with the metric.
            imgs_type: str
                Should be one of ["RAW", "DENOISED", "RAW_NOISY", "NOISY_DENOISED"].
        """
        types=["RAW", "DENOISED", "RAW_NOISY", "NOISY_DENOISED"]
        
        if not metric in metrics.keys():
            raise AttributeError("The input metric should be part of the usable metric. Verify the metrics_loading.py file.")
        if not imgs_type in types:
            raise AttributeError('The input imgs_type should be one of ["RAW", "DENOISED", "RAW_NOISY", "NOISY_DENOISED"].')
            
        self.metrics[metric][imgs_type]=score
        
    def add_time_spent(self, time):
        """
        Adds the time to the report
        """
        self.time = time
        
    def generate(self, filename="report", path=None, as_string=False):
        """
        Generates the text version of the report.
        
        Parameters:
            filename: str (default='report')
                Name to give to the exported report text file.
            path: str (default=None)
                The path to the directory to save the report in.
            as_string: bool (default=False)
                Indicates wether to return a string version of the report instead of saving it.
        """
        
        if path is None and not as_string:
            raise AttributeError("Please either indicate the path or set the as_string parameter.")
            
        nl = '\n'
        dnl = '\n\n'
        report = ""
        
        # Adding title
        report += self.title + dnl
        
        # Adding Datasets
        report += self.datasets_title +nl 
        report += nl.join(self.datasets)
        report += dnl
        
        # Adding tasks used
        report += self.tasks_title + nl
        report += nl.join(list(set(self.tasks_models.values())))
        report += dnl
        
        # Adding tasks models
        report += self.task_models_title + nl
        report += nl.join([ f"{k:20} [{self.tasks_models[k]}]" for k in self.tasks_models.keys()])
        report += dnl
        
        # Adding time spent
        report += self.time_title + nl
        report += self.time
        report += dnl
        
        # Adding metrics results
        report += self.metrics_title + nl
        for m in self.metrics.keys():
            report += self.metrics_prefix + ' ' + m + nl
            for t in self.metrics[m].keys():
                report += f"{t:20}{self.metrics[m][t]}"+nl
            report += nl
        report += nl
        
        if as_string:
            return report
        
        else:
            if not os.path.isdir(path):
                print(f"No folder found at {path}. Creating one to save results.")
                os.mkdir(path)
            filename = path + filename + '_' + datetime.now().strftime("%d_%m_%Y-%H_%M_%S") + ".txt"
            with open(filename, 'w') as output_file:
                output_file.write(report)

            print(f"Successfully exported the report to {filename}.")
        
        