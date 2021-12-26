## CS-503 Visual Intelligence
# Benchmarking Framework for Bad Weather Distortion in Semantic Segmentation in the Context of Autonomous Driving

## Proposal Recap
A benchmarking framework proposing a standard dataset and procedure to assess the performance of bad weather denoising models in an autonomous driving scenario. The aim is to provide an accessible framework that proposes to evaluate denoising models on a special road dataset. The user should input a candidate denoising model and our framework will perform tests with various metrics to assess performance of the candidate model.

This repository contains:
- requirements for the pipeline steps under `requirements/` folder.
- A sample candidate model in the folder `src/`
- Multiple segmentation models in the folders `sota_model/HRNet` and `sota_model/models`
- Metrics in the `sota_model/metrics` folder

<!-- - distorted (bad weather condition) image dataset,
- state-of-art model to perform evaluation task (semantic segmentation or object detection),
- description of the methods that a candidate model needs to offer to use the framework,
- pipeline to get metrics about the performance of a candidate model. -->



### Resources

<!-- Remove this? -->

Below the resources used for the development of the project:
- Project Proposal [Google Doc](https://docs.google.com/document/d/1qNOLPn8raD1vMe1DtMaB38gz32ec2g34F1GkgsOPekA/edit#)

## Milestone 1
[Milestone 1 report](https://www.overleaf.com/project/618069eb5c0c60d2b127609f)

## Milestone 2
[Milestone 2 report](https://www.overleaf.com/read/zsdgtwrbpnxf)

## Final report
[Final report](https://www.overleaf.com/read/csphjyvbpbsr)

# Credits for code parts

## Syn2Real

The contents of the folder `candidate_model/Syn2Real` originates from [the Syn2Real repository](https://github.com/rajeevyasarla/Syn2Real)
(version of the 11th of December 2021) and belongs exclusively to its authors. It is and should be used according to the License under which the code is shared.
Minor modifications were made by us in the `test.py` file to be able to run the code on Windows. We also added the file
`run-syn2real.py` to be able to test our pipeline. Otherwise everything is untouched, including the README.md file
inside the folder.

## HRNet

The contents of the folder `sota_model/HRNet` originates from [the HR-Net repository](https://github.com/HRNet/HRNet-Semantic-Segmentation)
(version of the 11th of December 2021) and belongs exculsively to its authors. It is and should be used according to the License under which the code is shared.
Small modifications were made by us for the experiments, the datasets and other various adaptive modifications.

For the models used in the framework under `sota_model/models`, please refer to the [Segmentation Models for Pytorch](https://smp.readthedocs.io/en/latest/) documentation, and the [PytorchCV](https://pypi.org/project/pytorchcv/) documentation.
