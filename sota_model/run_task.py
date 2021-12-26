import argparse

from models_loading import *
from data_loading import *
from tasks_loading import *
from metrics_loading import *
from time_loading import *

from utils import *

import torch


class TaskRunner:

    def __init__(self, args):

        self.model = args.fm  # Framework model
        self.dataset = args.d  # Dataset to use
        self.task = args.t  # Task to perform

        # Note: asserting values passed are valid is already done in run.py

        self.denoised_dir = "../candidate_model_predictions"  # Directory of candidate model predictions
        self.output_dir = "../output/"  # Directory containing all outputs
        self.output_images_dir = ["clean_seg/", "noisy_seg/", "denoised_seg/"]

        # Create required directories
        os.makedirs(self.output_dir, exist_ok=True)
        for d in self.output_images_dir:
            os.makedirs(os.path.join(self.output_dir, d), exist_ok=True)

        # Set attributes according to requested dataset
        if self.dataset == "cityscapes":
            self.dataset_name = "Cityscapes"
            self.dataset_dir = "HRNet/data/cityscapes"
            self.dataset_distorted = ""  # TODO
        else:
            raise ValueError("Dataset value passed invalid, verify run.py assertions...")

        if self.task == "semantic":
            self.metrics = ["iou", "dice"]
        else:
            raise ValueError("Task value passed invalid, verify run.py assertions...")

    def run_task(self):

        # Start timer
        start = start_timer()

        # TODO: Load data from dataset & predictions from candidate model
        image_set_names = ["CLEAN", "NOISY", "CLEAN_DENOISED", "DENOISED"]
        # TODO assert length
        image_sets = dict()
        for name in image_set_names:
            # Fake data for testing purposes
            image_sets[name] = torch.randn(1, 3, 1024, 2048)

        # Load model
        model = load_model(self.model).eval()

        # Load task
        task = load_tasks(self.task)
        task_model = task(model)

        # Run model on all data
        results = dict()

        for name in image_set_names:
            results[name] = task_model(image_sets[name], get_colored=True, get_classification=True)

        # Evaluate and store results
        evaluations = dict()
        for m in self.metrics:
            metric = load_metrics(m)
            metric_evals = dict()
            for i, name in enumerate(image_set_names):
                y, colored, classified = results[name]

                store_results(colored, os.path.join(self.output_dir, self.output_images_dir[i]))  # works until here
                metric_evals[name] = metric(y, torch.from_numpy(classified))
            evaluations[m] = metric_evals

        # Stop timer
        end = end_timer()
        time_spent = time_for_report(start, end)

        # Generate report
        report = Report()

        report.add_dataset(self.dataset_name)
        report.add_task_model(self.model, self.task)
        report.add_time_spent(time_spent)

        for m in self.metrics:
            for name in image_set_names:
                report.add_metric(m, evaluations[m][name], name)

        report.generate(path=os.path.join(self.output_dir, 'reports/'))


if __name__ == '__main__':
    # Read inputs
    parser = argparse.ArgumentParser(description='Parameters for running the segmentation task.')
    parser.add_argument("-fm", help="Framework model to use", type=str)  # default values already defined in run.py
    parser.add_argument("-d", help="Dataset to use", type=str)
    parser.add_argument("-t", help="Task to perform by the framework", type=str)
    args = parser.parse_args()

    runner = TaskRunner(args)
    runner.run_task()
