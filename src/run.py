import os
import subprocess
import argparse


class Runner:

    def __init__(self, args):
        # set arguments as attributes of the class
        self.candidate = args.m
        self.run_file = args.f
        self.framework_model = args.fm
        self.dataset = args.d
        self.task = args.t

        # Run some assertions:
        # - Args without default value exist
        # - Folder with model name exists inside folder candidate_model
        # - Requirements file corresponding to the candidate model exists
        # - Runfile provided for candidate model exists inside model folder
        # - Task is valid
        # - Framework model passed is valid
        # - Dataset passed is valid
        # - Framework model and dataset requested suitable for task
        #   -> add further checks inplace of the 'False' if more tasks are supported by the framework

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
        assert \
            self.task == "semantic" and self.framework_model in {"UNet", "SiNet"} and self.dataset in {"cityscapes"}\
            or False, "Framework model and/or dataset passed are not suitable for the requested task."

        # Run requirement file for pipeline
        run_req = subprocess.run(
            ["pip", "install", "-r", "../requirements_files/requirements.txt"], shell=True, stdout=subprocess.DEVNULL)
        assert run_req.returncode == 0, "Could not install requirements for benchmarking framework."

        # Create required directories
        os.makedirs("../candidate_model_predictions/", exist_ok=True)


    def run_candidate_model_on_dataset(self):

        # Run requirements file for candidate model
        run_req_candidate = subprocess.run(
            ["pip", "install", "-r", os.path.join("../requirements_files/", self.candidate + "-requirements.txt")], shell=True,
            stdout=subprocess.DEVNULL)
        assert run_req_candidate.returncode == 0, "Could not install requirements for candidate model."

        # TODO: Handle different datasets
        # Set path to dataset images - Paths passed are specified w.r.t. dir repo_root/src because we call the commands from there, meaning the paths are always relative to this location
        dataset_dir = "../placeholder_dataset_synthetic/"  # To be changed to ../../dataset/ when our dataset images will be ready

        # Set path to output directory - Paths passed are specified w.r.t. dir repo_root/src because we call the commands from there, meaning the paths are always relative to this location
        output_dir = "../candidate_model_predictions/"

        # Run python file specified in args with the two paths as arguments
        run_candidate = subprocess.run(["python", os.path.join("../candidate_model/", self.candidate, self.run_file), "-d", dataset_dir, "-o", output_dir], shell=True)#, stdout=subprocess.DEVNULL)
        assert run_candidate.returncode == 0, "An error occurred while running the candidate model."


    def run_framework_task(self):

        # Run requirements file for framework model
        run_req = subprocess.run(
            ["pip", "install", "-r", "../requirements_files/requirements-" + self.task + ".txt"],
            shell=True, stdout=subprocess.DEVNULL)
        assert run_req.returncode == 0, "Could not install requirements for framework model."

        # Run runfile for framework model
        run_task = subprocess.run(["python", "../sota_model/run_task.py", "-fm", self.framework_model, "-d", self.dataset, "-t", self.task], shell=True)
        assert run_task.returncode == 0, "An error occurred while running the framework model."


if __name__ == '__main__':
    # Read inputs from user when calling the run.py file
    parser = argparse.ArgumentParser(description='Parameters for running the benchmarking framework.')
    parser.add_argument("-m", help="Candidate model name", type=str)
    parser.add_argument("-f", help="File to run to obtain predictions", type=str)
    parser.add_argument("-fm", help="Framework model to use", type=str, default="UNet")
    parser.add_argument("-d", help="Dataset to use", type=str, default="cityscapes")
    parser.add_argument("-t", help="Task to perform by the framework", type=str, default="semantic")
    parser.add_argument("-l", help="List tasks, models and datasets supported by the framework. Pass 't' for tasks,"
                                   "'fm' for models and 'd' for datasets.")
    args = parser.parse_args()

    if args.l is not None:
        assert args.l in {"t", "fm", "d"}, "Invalid value for listing"
        # TODO: List available tasks, models for each task and datasets for each task in the console, finish process
        pass
    else:
        runner = Runner(args)
        runner.run_candidate_model_on_dataset()
        runner.run_framework_task()
