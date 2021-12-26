# Instructions to run the benchmarking framework

## Pre-requisites for your model

1. A **requirements file**: A txt file containing all requirements of modules with their versions for the model to run properly.
Follow the naming convention `*-requirements.txt` where * is the exact name of your candidate model to avoid errors.
2. A **python run file** that takes 1. the path to the directory of the benchmarking framework dataset, 
   and 2. a path to an output directory as inputs. It should preprocess the images in the dataset, then predict with the
   candidate model and outputs the denoised images inside the specified output directory. This file will be called with the following arguments:
   `python <your_run_file.py> -d <benchmarking_dataset_directory> -o <output_directory>`, so make sure it can read these arguments.

## Place your files

1. Create a folder with your model's name inside the folder `candidate_model`.
2. Place the **python run file**, your trained model and all other files needed to run your candidate model inside this folder.
3. Place your **requirements file** inside the `requirements_files` folder.
4. Download the dataset [TODO: Add explanation on where and how to download the dataset and place files]

## Run the framework

1. Open a terminal.
2. Navigate to the root of this repository.
3. Set the python path to the absolute path to the root of this repository with
`export PYTHONPATH=<project_path>` (linux) or directly in the environment variables
4. Run the command `cd src && python run.py -m <exact_candidate_model_name> -f <run_file_name.py>`
5. Add further arguments to this command if needed:
   * `-fm <framework-model>` to use a chosen framework model, default is `Unet`
   * `-d <dataset-name>` to use a chosen dataset, default is `cityscapes`
   * `-t <task-name>` to perform a chosen task, default is `semantic`
   * `-n <noise-type>` to use a different noise type, default is `rain`
6. Use `python run.py -l <'t'/'fm'/'d'>` to list supported tasks (use `t`), framework models (use `fm`) or datasets (use `d`)

## See the results

The segmented images and the report are all available inside the `output` folder, generated during execution of the framework.
