# MS-SKResNet for AFB1 Content-Grade Classification in Rice

This repository provides the implementation of the Multi-Scale Selective Kernel Residual Network (MS-SKResNet) developed for four-class aflatoxin B1 (AFB1) content-grade classification in mold-contaminated rice using near-infrared hyperspectral imaging (NIR-HSI).

The code corresponds to the computational workflow described in the revised manuscript:

**Classification of Aflatoxin B1 Content Grades in Mold-Contaminated Rice Using Near-Infrared Hyperspectral Imaging Integrated with a Multi-Scale Selective Kernel Residual Network**

---

## Overview

The repository contains the main computational workflow used for MS-SKResNet training and evaluation, including:

- repeated stratified sample partitioning;
- spectral preprocessing;
- multiplicative random-scaling augmentation;
- MS-SKResNet model construction;
- training-validation based training-duration selection;
- final model fitting using the complete calibration set;
- prediction-set evaluation;
- calculation of classification metrics and confusion matrices.

The workflow was designed so that the prediction samples remain excluded from preprocessing-parameter estimation, data augmentation, model selection, and model fitting.

---

## Repository Files

### `MS_SKResNet.py`

Implementation of the proposed MS-SKResNet architecture, including:

- multi-scale spectral feature fusion using parallel convolutional branches;
- selective-kernel adaptive feature weighting;
- residual feature propagation;
- global average pooling and global max pooling;
- fully connected four-class classifier.

### `train10.py`

Main training and evaluation script used for the repeated experiments reported in the manuscript. The script implements:

- 10 repeated stratified random splits;
- training, validation, and prediction subset construction;
- spectral preprocessing;
- multiplicative random-scaling augmentation;
- training-duration selection using the validation subset;
- final refitting using the complete calibration set;
- prediction on original, non-augmented samples;
- calculation of accuracy, Macro-Precision, Macro-Recall, and Macro-F1;
- generation of confusion matrices and result files.

`MS_SKResNet.py` and `train10.py` should be placed in the same directory.

---

## Dataset Partitioning

The study dataset contains 264 original rice samples.

For each of the 10 repeated stratified evaluations:

- 184 samples are assigned to the calibration set;
- 80 samples are assigned to the prediction set;
- the 184 calibration samples are initially divided into:
  - 147 training samples;
  - 37 validation samples.

Random seeds 42–51 are used for the 10 repeated evaluations.

During the training-validation stage, the validation subset is used to determine the training duration.

After the training duration has been selected, a newly initialized model is fitted using all 184 original calibration samples together with their corresponding augmented spectra.

The final model is then evaluated using the corresponding 80 original prediction samples.

The prediction samples are not used for training, validation, preprocessing-parameter estimation, augmentation, checkpoint selection, or training-duration selection.

---

## Spectral Preprocessing and Data Augmentation

The default workflow reported in the manuscript uses:

1. second-derivative (D2) spectral preprocessing;
2. multiplicative random-scaling augmentation.

For multiplicative random scaling, each processed fitting spectrum is multiplied by five independently sampled scaling factors from a uniform distribution between 0.9 and 1.1.

Data augmentation is applied only to samples used for model fitting.

Validation and prediction samples remain original and unaugmented.

---

## AFB1 Content-Grade Definition

The four AFB1 content grades are defined using the assay-derived AFB1 concentrations:

- **Class I:** 1 ≤ AFB1 < 5 μg/kg
- **Class II:** 5 ≤ AFB1 < 10 μg/kg
- **Class III:** 10 ≤ AFB1 < 20 μg/kg
- **Class IV:** 20 ≤ AFB1 ≤ 40 μg/kg

The 10 μg/kg boundary corresponds to the regulatory threshold considered in the study.

---

## Model Training Settings

The principal MS-SKResNet training settings are:

- Optimizer: AdamW
- Initial learning rate: 0.001
- Weight decay: 0.01
- Batch size: 16
- Maximum epochs: 200
- Early-stopping patience: 80 epochs
- Learning-rate schedule: cosine annealing
- Minimum learning rate: 1 × 10⁻⁵
- Loss function: class-weighted cross-entropy
- Maximum gradient norm: 1.0
- Number of augmented spectra per original fitting spectrum: 5
- Multiplicative scaling range: 0.9–1.1

During the training-validation stage, the checkpoint with the highest validation accuracy is used to determine the training duration. The selected number of epochs is then used for final fitting with the complete 184-sample calibration set.

---

## Input Data Format

The training script accepts a CSV file containing the assay-derived AFB1 concentrations and the corresponding NIR spectra.

The expected structure is:

- the **first row**, from the second column onward, contains wavelength values;
- the **first column**, from the second row onward, contains the assay-derived AFB1 concentration of each sample;
- each subsequent row corresponds to one rice sample;
- the remaining columns contain the corresponding spectral variables.

Conceptually, the CSV structure is:

| AFB1 concentration | Wavelength 1 | Wavelength 2 | ... | Wavelength 196 |
|---|---:|---:|---:|---:|
| wavelength header | λ1 | λ2 | ... | λ196 |
| Sample 1 concentration | spectrum | spectrum | ... | spectrum |
| Sample 2 concentration | spectrum | spectrum | ... | spectrum |
| ... | ... | ... | ... | ... |

The dataset used in the study contained:

- 264 samples;
- 196 retained spectral variables.

Class labels are generated automatically from the assay-derived AFB1 concentrations according to the class boundaries defined above.

The wavelength values must be strictly monotonic, and the spectral values must contain no missing or non-finite values.

---

## Computational Environment

The implementation used in the study was developed under the following environment:

- Operating system: Windows 11
- Python: 3.13
- PyTorch: 2.9.0
- GPU: NVIDIA GeForce RTX 5060

Main Python dependencies include:

- NumPy
- Pandas
- scikit-learn
- Matplotlib
- joblib

Additional dependencies are required only for specific optional functions:

- `openpyxl` — Excel output;
- `SciPy` — Savitzky–Golay preprocessing;
- `PyWavelets` — wavelet preprocessing.

---

## Usage

Place `MS_SKResNet.py` and `train10.py` in the same directory.

Run the default workflow using:

```bash
python train10.py --csv path/to/data.csv
