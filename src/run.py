import os
import subprocess
import sys
import argparse


class Runner:

    def __init__(self, args):
        # set arguments as attributes of the class
        self.candidate = args.m
        self.run_file = args.f

        # Run requirement files
        ## requirements.txt
        run_req = subprocess.run(["pip", "install", "-r", "../requirements_files/requirements.txt"], shell=True, stdout=subprocess.DEVNULL)
        assert run_req.returncode == 0, "Could not install requirements for benchmarking framework."
        ## [Candidate_model_name]-requirements.txt
        run_req_candidate = subprocess.run(["pip", "install", "-r", "../requirements_files/"+ self.candidate + "-requirements.txt"], shell=True, stdout=subprocess.DEVNULL)
        assert run_req_candidate.returncode == 0, "Could not install requirements for candidate model."

        # Create required directories
        os.makedirs("../candidate_model_predictions/", exist_ok=True)


    def run_candidate_model_on_dataset(self):
        # Set path to dataset images - Paths passed are specified w.r.t. dir repo_root/src because we call the commands from there, meaning the paths are always relative to this location
        dataset_dir = "../placeholder_dataset_synthetic/"  # To be changed to ../../dataset/ when our dataset images will be ready

        # Set path to output directory - Paths passed are specified w.r.t. dir repo_root/src because we call the commands from there, meaning the paths are always relative to this location
        output_dir = "../candidate_model_predictions/"

        # Run python file specified in args with the two paths as arguments
        run_candidate = subprocess.run(["python", os.path.join("../candidate_model/", self.candidate, self.run_file), "-d", dataset_dir, "-o", output_dir], shell=True)#, stdout=subprocess.DEVNULL)
        assert run_candidate.returncode == 0, "An error occurred while running the candidate model."

        # Python files needed - add to userguide
        #  - A python file that takes the directory of the raw dataset images and a path to an output directory as
        #    input, then preprocesses the dataset images, then predicts with the NN and outputs the denoised images
        #    inside the specified directory
        #     - Arguments should be -d <dataset_images_dir> -o <output_dir>
        #     - Paths passed are specified w.r.t. dir repo_root/src because we call the commands from there, meaning the paths are always relative to this location


    def run_sota_docker(self):
        # Start docker for SOTA
        pass


if __name__ == '__main__':
    # Read inputs from user when calling the run.py file
    parser = argparse.ArgumentParser(description='Parameters for running the benchmarking framework.')
    parser.add_argument("-m", help="Candidate model name", type=str)
    parser.add_argument("-f", help="File to run to obtain predictions", type=str)
    args = parser.parse_args()

    runner = Runner(args)
    runner.run_candidate_model_on_dataset()
    runner.run_sota_docker()
