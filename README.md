# MS-SKResNet for AFB1 Classification in Rice

This repository provides the implementation of a multi-scale selective kernel residual network (MS-SKResNet) for four-class aflatoxin B1 (AFB1) content grade classification in rice using near-infrared hyperspectral imaging (NIR-HSI).

## Overview

The code implements the complete workflow for AFB1 content grade classification, including:

- NIR spectral preprocessing
- Data augmentation
- MS-SKResNet model construction
- Repeated stratified training and evaluation
- Classification performance analysis

## Files

### MS_SKResNet.py

Implementation of the MS-SKResNet architecture, including:

- Multi-scale convolution modules
- Selective kernel blocks
- Residual learning structure
- Classification network

### train10.py

Training and evaluation pipeline including:

- 10 repeated stratified random splits
- Spectral preprocessing
- Data augmentation
- Model training
- Performance evaluation

## Method

The workflow consists of:

1. Spectral preprocessing using second-order derivative (D2)
2. Data augmentation on training samples
3. MS-SKResNet model training
4. Four-class AFB1 content grade classification

## Evaluation Metrics

The classification performance was evaluated using:

- Accuracy
- Macro-Precision
- Macro-Recall
- Macro-F1
- Confusion matrix

## Requirements

Python 3.x

Main dependencies:

- PyTorch
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib

## Citation

If you use this code, please cite the corresponding publication.
