## CS-503 Visual Intelligence
# Benchmarking Framework for Bad Weather Distortion in Object Detection / Semantic Segmentation in the Context of Autonomous Driving

## Proposal Recap
A benchmarking framework proposing a standard dataset and procedure to assess the performance of bad weather denoising models in an autonomous driving scenario. The aim is to provide an accessible framework that proposes to evaluate denoising models on a special road dataset. The user should input a candidate denoising model and our framework will perform tests with various metrics to assess performance of the candidate model.

This repository contains:
- distorted (bad weather condition) image dataset,
- state-of-art model to perform evaluation task (semantic segmentation or object detection),
- description of the methods that a candidate model needs to offer to use the framework,
- pipeline to get metrics about the performance of a candidate model.

### Resources

Below the resources used for the development of the project:
- Project Proposal [Google Doc](https://docs.google.com/document/d/1qNOLPn8raD1vMe1DtMaB38gz32ec2g34F1GkgsOPekA/edit#)

## Milestone 1
[Milestone 1 report](https://www.overleaf.com/project/618069eb5c0c60d2b127609f)

## Milestone 2


# Credits for code parts

## Syn2Real

The entire content of the folder `candidate_model/Syn2Real` was taken from [this repository](https://github.com/rajeevyasarla/Syn2Real)
(version of the 11th of December 2021) and belongs exclusively to their authors. Small modification were made by us in 
the `test.py` file to be able to run the code on Windows. We also wrote and added the file `run_syn2real.py`, which is the file required
by us to test a model with our benchmarking framework (see userguide.md for more information). Otherwise everything is untouched, 
including the README.md file inside the folder.
