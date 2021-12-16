# High-resolution networks and Segmentation Transformer for Semantic Segmentation

## Before running the model

The model is ran using pre-configured experiments available under `experiments/`.

To run the model, we use a pretrained file located under `pretrained_models/`. The files end with the `.pth` extension.

In this project we use :
- HRNetV2-W48 + OCR pretrained ([available here](https://github.com/HRNet/HRNet-Semantic-Segmentation), `hrnet_ocr_cs_8162_torch11.pth`)

The cityscapes dataset is used by this model.
It can be found at the [official website](https://www.cityscapes-dataset.com/).


The data is stored under `../../data/` directory.
It is organized as follows
````
$ROOT
├── data
│   └── cityscapes
│       ├── gtFine
│       │   ├── test
│       │   ├── train
│       │   └── val
│       └── leftImg8bit
│           ├── test
│           ├── train
│           └── val
└── sota_model
    └── HRNet
        └── pretrained_models
            └── hrnet_ocr_cs_8162_torch11.pth
````

## Running the model

The model necessitates 16GB of RAM (8GB of RAM not enough).

The repo does not contain the following files that need to be downloaded:


The command to run the model is the following : <br>
`python tools/test.py --cfg experiments/cityscapes/seg_hrnet_ocr_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 TEST.FLIP_TEST True`
