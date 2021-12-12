import os
import subprocess
import sys
import getopt


class Runner:

    def __init__(self, args):
        # parse arguments, raise exceptions if not complete
        # set arguments as attributes of the class
        self.args = args  # Replace with nice attribution after parsing

    def run_candidate_model_on_dataset(self):
        # Set path to dataset images
        # Set path to output directory
        # Run python file specified in args with the two paths as arguments

        # Python files needed
        #  - A python file that takes the directory of the raw dataset images and a path to an output directory as
        #    input, then preprocesses the dataset images, then predicts with the NN and outputs the denoised images
        #    inside the specified directory
        #     - Arguments should be -d <dataset_images_dir> -o <output_dir>
        #     - Paths passed are specified w.r.t. dir repo_root/candidate_model/model_name/
        pass

    def run_sota_docker(self):
        # Start docker for SOTA
        pass


def run():
    try:
        # Read inputs from user when calling the run.py file
        runner = Runner(sys.argv)
        runner.run_candidate_model_on_dataset()
        runner.run_sota_docker()
    # Add catches for specifically defined errors here if any
    except getopt.GetoptError as e:
        print("run.py -m <model_name> -f <file_with_functions>")
    except BaseException as e:
        print("An unidentified error occurred:")
        print(e)


def sandbox():  # remove when done with experiments
    # Run commands
    test_cd = os.system("cd ../sota_model")
    print("The command 'cd ../sota_model' with os.system ran with exit code", test_cd)
    list_files = subprocess.run(["cd", "../sota_model"], shell=True, stdout=subprocess.DEVNULL)  # Recommended version
    print("The command 'cd ../sota_model' with subprocess.run ran with exit code", list_files.returncode)

    # Check OS

    # Get user arguments
    print("Number of arguments:", len(sys.argv))
    print("Arguments list:", str(sys.argv))


if __name__ == '__main__':

    #run()
    #sandbox()
    pass
