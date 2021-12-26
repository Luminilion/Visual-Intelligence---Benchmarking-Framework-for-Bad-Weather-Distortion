import os
import subprocess
import argparse
import glob

import sys
sys.path.insert(1, "../")

from sota_model.models_loading import *
from sota_model.data_loading import *
from sota_model.tasks_loading import *

class Runner:

    def __init__(self, args):
        # set arguments as attributes of the class
        self.candidate = args.m
        self.run_file = args.f
        self.framework_model = args.fm
        self.dataset = args.d
        self.task = args.t
        self.noise_type = args.n

        # Note: Paths passed here are specified w.r.t. dir repo_root/src because we call the commands from there,
        # meaning the paths are always relative to this location

        # Run some assertions:
        # - Args without default value exist
        # - Folder with model name exists inside folder candidate_model
        # - Requirements file corresponding to the candidate model exists
        # - Runfile provided for candidate model exists inside model folder
        # - Task is valid
        # - Framework model passed is valid
        # - Dataset passed is valid
        # - Noise type is valid
        # - Framework model and dataset requested suitable for task
        #   -> add further checks inplace of the 'False' if more tasks are supported by the framework
        # - Dataset directory for clean images is available
        # - Dataset directory for noisy images is available

        assert self.candidate is not None, "Provide the name of your candidate model with '-m <candidate-model-name>'!"
        assert self.run_file is not None, \
            "Provide the name of the runfile for your candidate model with '-f <runfile-name.py>'"
        assert os.path.isdir(os.path.join("../candidate_model/", self.candidate)), \
            "Could not locate your model's folder inside the folder 'candidate_model'!"
        assert os.path.isfile(os.path.join("../requirements_files/", self.candidate + "-requirements.txt")), \
            "Could not locate the requirements file for your model inside the 'requirements_files' folder!"
        assert os.path.isfile(os.path.join("../candidate_model/", self.candidate, self.run_file)), \
            "Could not locate the runfile of your candidate model inside the folder of your model!"
        assert self.task in {"semantic"}, "Invalid task requested."
        assert self.framework_model in {"UNet", "SiNet"}, "Pass a valid framework model."
        assert self.dataset in {"cityscapes"}, "Pass a valid dataset name."
        assert self.noise_type in {"rain", "fog"}, "Pass a valid noise type."
        assert \
            self.task == "semantic" and self.framework_model in {"UNet", "SiNet"} and self.dataset in {"cityscapes"}\
            or False, "Framework model and/or dataset passed are not suitable for the requested task."
        assert os.path.isdir(os.path.join("../dataset/", self.dataset +"/")), \
            "Could not locate clean dataset files. Please download and place them as indicated in the userguide!"
        assert os.path.isdir(os.path.join("../dataset/", "weather_" + self.dataset + "/")), \
            "Could not locate noisy dataset files. Please download and place them as indicated in the userguide!"

        # Run requirement file for pipeline
        run_req = subprocess.run(
            ["pip", "install", "-r", "../requirements_files/requirements.txt"], shell=True, stdout=subprocess.DEVNULL)
        assert run_req.returncode == 0, "Could not install requirements for benchmarking framework."

    def __get_all_subfolders_and_run_candidate_model(self, dataset_dir, output_dir):

        # Additional path specifications depending on the exact dataset
        if self.dataset == "cityscapes":
            additional_part = "leftImg8bit/"
            dataset_dir = os.path.join(dataset_dir, additional_part)
            output_dir = os.path.join(output_dir, additional_part)

        dataset_subdirs = glob.glob(os.path.join(dataset_dir, "**"), recursive=True)
        dataset_subdirs = set([os.path.split(p)[0][len(dataset_dir):] for p in dataset_subdirs if os.path.splitext(p)[
            1]])  # Split head and file, keep head only if entry has an extension (i.e. is a file), strip dataset_dir

        # Iterate over subfolders
        for subdir in dataset_subdirs:
            current_dataset_subdir = os.path.join(dataset_dir, subdir)
            current_output_subdir = os.path.join(output_dir, subdir)

            os.makedirs(current_output_subdir, exist_ok=True)  # Create directories if they don't already exist

            # Run python file specified in args with the two paths as arguments
            run_candidate = subprocess.run(
                ["python", os.path.join("../candidate_model/", self.candidate, self.run_file), "-d",
                 current_dataset_subdir, "-o", current_output_subdir],
                shell=True)  # We want the console output of the candidate model runfile, in case the user needs it
            assert run_candidate.returncode == 0, \
                "An error occurred while running the candidate model on images in directory " + current_dataset_subdir

    def run_candidate_model_on_dataset(self):

        # Run requirements file for candidate model
        run_req_candidate = subprocess.run(
            ["pip", "install", "-r", os.path.join("../requirements_files/", self.candidate + "-requirements.txt")], shell=True,
            stdout=subprocess.DEVNULL)
        assert run_req_candidate.returncode == 0, "Could not install requirements for candidate model."

        # Set path to dataset images and output directory
        # - Clean
        dataset_dir = os.path.join("../dataset/", self.dataset + "/")
        output_dir = os.path.join("../candidate_model_predictions/", self.dataset + "/")
        # - Noisy
        prefix = "weather_"
        weather_dataset_dir = os.path.join("../dataset/", prefix + self.dataset + "/")
        weather_output_dir = os.path.join("../candidate_model_predictions/", prefix + self.dataset + "/")

        self.__get_all_subfolders_and_run_candidate_model(dataset_dir, output_dir)
        self.__get_all_subfolders_and_run_candidate_model(weather_dataset_dir, weather_output_dir)


    def run_framework_task(self):

        # Run requirements file for framework model
        run_req = subprocess.run(
            ["pip", "install", "-r", "../requirements_files/requirements-" + self.task + ".txt"],
            shell=True, stdout=subprocess.DEVNULL)
        assert run_req.returncode == 0, "Could not install requirements for framework model."

        # Run runfile for framework model
        run_task = subprocess.run(
            ["python", "../sota_model/run_task.py", "-fm", self.framework_model, "-d", self.dataset, "-t", self.task,
             "-n", self.noise_type],
            shell=True)
        assert run_task.returncode == 0, "An error occurred while running the framework model."


if __name__ == '__main__':
    # Read inputs from user when calling the run.py file
    parser = argparse.ArgumentParser(description='Parameters for running the benchmarking framework.')
    parser.add_argument("-m", help="Candidate model name", type=str)
    parser.add_argument("-f", help="File to run to obtain predictions", type=str)
    parser.add_argument("-fm", help="Framework model to use", type=str, default="UNet")
    parser.add_argument("-d", help="Dataset to use", type=str, default="cityscapes")
    parser.add_argument("-t", help="Task to perform by the framework", type=str, default="semantic")
    parser.add_argument("-n", help="Noise type", type=str, default="rain")
    parser.add_argument("-l", help="List tasks, models and datasets supported by the framework. Pass 't' for tasks,"
                                   "'fm' for models and 'd' for datasets.")
    args = parser.parse_args()

    if args.l is not None:
        assert args.l in {"t", "fm", "d"}, "Invalid value for listing"
        if args.l == "t":
            load_tasks("", list_available=True)
        elif args.l == "fm":
            load_model("", list_available=True)
        else:
            load_data("", list_available=True)
    else:
        runner = Runner(args)
        runner.run_candidate_model_on_dataset()
        runner.run_framework_task()
