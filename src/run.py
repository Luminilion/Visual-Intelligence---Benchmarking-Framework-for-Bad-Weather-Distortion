import os
import subprocess
import argparse


class Runner:

    def __init__(self, args):
        # set arguments as attributes of the class
        self.candidate = args.m
        self.run_file = args.f

        # Run some assertions
        # - Args exist
        assert self.candidate is not None, "Provide the name of your candidate model with '-m <candidate-model-name>'!"
        assert self.run_file is not None, \
            "Provide the name of the runfile for your candidate model with '-f <runfile-name.py>'"
        # - Folder with model name exists inside folder candidate_model
        assert os.path.isdir(os.path.join("../candidate_model/", self.candidate)), \
            "Could not locate your model's folder inside the folder 'candidate_model'!"
        # - Requirements file corresponding to the candidate model exists
        assert os.path.isfile(os.path.join("../requirements_files/", self.candidate + "-requirements.txt")), \
            "Could not locate the requirements file for your model inside the 'requirements_files' folder!"
        # - Runfile provided for candidate model exists inside model folder
        assert os.path.isfile(os.path.join("../candidate_model/", self.candidate, self.run_file)), \
            "Could not locate the runfile of your candidate model inside the folder of your model!"

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

        # Set path to dataset images - Paths passed are specified w.r.t. dir repo_root/src because we call the commands from there, meaning the paths are always relative to this location
        dataset_dir = "../placeholder_dataset_synthetic/"  # To be changed to ../../dataset/ when our dataset images will be ready

        # Set path to output directory - Paths passed are specified w.r.t. dir repo_root/src because we call the commands from there, meaning the paths are always relative to this location
        output_dir = "../candidate_model_predictions/"

        # Run python file specified in args with the two paths as arguments
        run_candidate = subprocess.run(["python", os.path.join("../candidate_model/", self.candidate, self.run_file), "-d", dataset_dir, "-o", output_dir], shell=True)#, stdout=subprocess.DEVNULL)
        assert run_candidate.returncode == 0, "An error occurred while running the candidate model."


    def run_segmentation(self):

        # Run requirements file for segmentation model

        # Run runfile for sgmentation model

        pass


if __name__ == '__main__':
    # Read inputs from user when calling the run.py file
    parser = argparse.ArgumentParser(description='Parameters for running the benchmarking framework.')
    parser.add_argument("-m", help="Candidate model name", type=str)
    parser.add_argument("-f", help="File to run to obtain predictions", type=str)
    args = parser.parse_args()

    runner = Runner(args)
    runner.run_candidate_model_on_dataset()
    runner.run_segmentation()
