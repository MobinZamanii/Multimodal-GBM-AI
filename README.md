# Multimodal-GBM-AI

A multimodal deep learning framework for brain tumor MRI analysis with transfer learning, GTR prediction, and survival modeling.

## Overview

This project aims to develop a multimodal deep learning framework for glioblastoma (GBM) analysis by integrating MRI imaging data with clinical information.

The framework follows a transfer-learning strategy in which an MRI encoder is first pretrained on the BraTS 2021 dataset and subsequently adapted to clinically oriented GBM datasets, including UPENN-GBM and UCSF-PDGM.

The final framework is designed to investigate two major clinical prediction tasks:

- Gross Total Resection (GTR) prediction
- Overall Survival prediction

## Research Pipeline


BraTS 2021
(T1, T1ce, T2, FLAIR + Segmentation)
                │
                ▼
       3D MRI Pretraining
                │
                ▼
          MRI Encoder
                │
        Transfer Learning
                │
        ┌───────┴────────┐
        ▼                ▼
    UPENN-GBM         UCSF-PDGM
        │                │
        └───────┬────────┘
                ▼
       Multimodal Learning
                │
        ┌───────┴────────┐
        ▼                ▼
       GTR            Survival
    Prediction        Prediction

Datasets
BraTS 2021

BraTS 2021 is used for MRI encoder pretraining.

The available training data contain four MRI modalities:

T1
T1ce
T2
FLAIR

along with tumor segmentation masks.

The current dataset contains 1,251 complete training cases.

BraTS is used only as a pretraining dataset in the current pipeline. Clinical metadata from BraTS are not required for this stage.

UPENN-GBM

UPENN-GBM provides multimodal MRI and clinical information for downstream GBM modeling.

The dataset is used for clinically oriented prediction tasks, including:

GTR prediction
Overall Survival prediction
UCSF-PDGM

UCSF-PDGM is intended as an additional GBM cohort for model development and evaluation, helping investigate model generalization across datasets.

Current Stage

The project is currently at the MRI pretraining stage.

A 3D segmentation network is being trained on BraTS 2021 to learn useful MRI representations.

The pretrained MRI encoder will subsequently be transferred to the downstream multimodal GBM framework.

Current Training Setup
Dataset: BraTS 2021
Complete cases: 1,251
Training cases: 1,001
Validation cases: 250
Training epochs: 30
Initial learning rate: 1e-4
GPU: NVIDIA GeForce GTX 1650
Framework: PyTorch
Medical imaging framework: MONAI
Pretraining Objective

The current pretraining stage uses tumor segmentation as the learning objective.

Primary metrics:

Dice coefficient
Intersection over Union (IoU)
Training loss
Validation loss

The purpose of this stage is to learn transferable MRI representations rather than to perform the final GTR or survival prediction.

Model Development

The planned architecture consists of several components:

MRI Encoder
     │
     ├──────────────┐
     │              │
Segmentation      Clinical
   Encoder         Encoder
     │              │
     └───────┬──────┘
             ▼
      Multimodal Fusion
             │
       ┌─────┴─────┐
       ▼           ▼
      GTR       Survival
     Head          Head

The MRI branch is designed as a 3D convolutional encoder.

Clinical and tumor-related information will be incorporated through dedicated feature encoders before multimodal fusion.

Project Structure
Multimodal-GBM-AI/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── configs/
├── notebooks/
│
├── results/
│   ├── figures/
│   └── logs/
│
└── checkpoints/
Technologies
Python
PyTorch
MONAI
NumPy
Pandas
Scikit-learn
NiBabel
Matplotlib
Seaborn
Reproducibility

The project is being developed with an emphasis on reproducible experiments.

Training configurations, preprocessing procedures, model definitions, evaluation scripts, and experiment results will be organized within the repository as the project progresses.

Large medical imaging datasets and trained model checkpoints are not included in this repository.

Research Goals

The main goals of this project are:

Learn transferable 3D MRI representations through BraTS pretraining.
Adapt the pretrained MRI encoder to GBM-specific datasets.
Integrate imaging and clinical information through multimodal learning.
Predict GTR status.
Model overall survival.
Evaluate generalization across independent GBM cohorts.
Project Status

Active Research — Under Development

Current stage:

 BraTS 2021 dataset preparation
 BraTS training/validation split
 3D MRI pretraining pipeline
 Tumor segmentation pretraining
 Complete BraTS pretraining
 Transfer pretrained MRI encoder
 UPENN-GBM downstream training
 UCSF-PDGM integration
 GTR prediction
 Survival modeling
 Cross-dataset evaluation
 Final analysis
Citation

If this project contributes to academic work, the corresponding publication and dataset references will be added here.

License

MIT License
