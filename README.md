# MM-DLS
# Multi-task deep learning based on PET/CT images for the diagnosis and prognosis prediction of advanced non-small cell lung cancer

## Overview

**MM-DLS** is a multi-modal, multi-task deep learning framework for the diagnosis, staging, and prognosis prediction of advanced non-small cell lung cancer (NSCLC). It integrates multi-source data including CT images, PET metabolic parameters, and clinical information to provide a unified, non-invasive decision-making tool for personalized treatment planning.

This repository implements the full MM-DLS pipeline, consisting of:
- Lung-lesion segmentation with cross-attention transformer
- Multi-modal feature fusion (CT, PET, Clinical)
- Multi-task learning: Pathological classification, TNM staging, DFS and OS survival prediction
- Cox proportional hazards survival loss

The framework supports both classification (adenocarcinoma vs squamous cell carcinoma) and survival risk prediction tasks, and has been validated on large-scale multi-center clinical datasets.

---

## Key Features

- **Multi-modal fusion:** Combines CT-based imaging features, PET metabolic biomarkers (SUVmax, SUVmean, SUVpeak, TLG, MTV), and structured clinical variables (age, sex, smoking status, smoking duration, smoking cessation history, tumor size).
- **Multi-task learning:** Simultaneous optimization for:
  - Histological subtype classification (LUAD vs LUSC)
  - TNM stage classification (I-II, III, IV)
  - Disease-free survival (DFS) prediction
  - Overall survival (OS) prediction
- **Attention-based feature fusion:** Transformer cross-attention module to integrate lung-lesion spatial information.
- **Survival modeling:** Incorporates Cox Proportional Hazards loss for survival time prediction.
- **Flexible data simulation and loading:** Includes utilities for synthetic data generation and multi-slice 2D volume processing.

---

## Architecture

The overall MM-DLS system consists of:

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Research-orange)

1. **Segmentation Module (LungLesionSegmentor):**
   - Shared ResNet encoder to extract features from CT images.
   - Dual decoders for lung and lesion segmentation.
   - Transformer-based cross-attention module for enhanced spatial feature interaction between lung and lesion regions.

2. **Feature Encoders:**
   - `LesionEncoder`: 2D convolutional encoder for lesion patches.
   - `SpaceEncoder`: 2D convolutional encoder for lung-space contextual patches.

3. **Attention Fusion Module:**
   - `LesionAttentionFusion`: Multi-head attention to fuse lesion and lung features into compact patient-level representations.

4. **Patient-Level Fusion Model (PatientLevelFusionModel):**
   - Fully connected network that combines imaging, PET, and clinical features.
   - Outputs classification logits, DFS and OS risk scores.

5. **Loss Functions:**
   - Binary cross-entropy loss for classification.
   - Cox proportional hazards loss (`CoxPHLoss`) for survival prediction.

---

## Code Structure

- `ModelLesionEncoder.py`: Lesion image encoder extracting discriminative features from multi-slice tumor regions.
- `ModelSpaceEncoder.py`: Lung space encoder modeling anatomical and spatial context beyond the lesion.
- `LesionAttentionFusion.py`: Attention-based fusion module for adaptive integration of lesion and spatial features.
- `ClinicalFusionModel.py`: Patient-level fusion network combining imaging features, radiomics, PET signals, and clinical variables.
- `HierMM_DLS.py`：Core hierarchical multimodal deep learning model supporting multi-task learning: (1)Subtype classification; (2)TNM stage prediction; (3)DFS and OS modeling
- `CoxphLoss.py`: Cox proportional hazards loss for survival modeling with censored data.
- `PatientDataset.py`:Patient dataset loader supporting imaging, radiomics, PET, clinical variables, survival outcomes, and treatment labels.
- `LungLesionSegmentation.py`: Lung-lesion segmentation model
- `ImageDataLoader.py`: Image preprocessing and loading utilities for multi-slice inputs.
- `plot_results.py`: Visualization utilities for Kaplan–Meier curves, hazard ratios, and survival analysis results.

---

## Data Format

The input data is organized per patient as follows:

### Imaging Data:
- CT slices (PNG format)
- Lung masks (binary masks, PNG)
- Lesion masks (binary masks, PNG)
- Slices grouped per patient ID

### Tabular Data:
- Radiomics features: 128-dimensional vector (PyRadiomics extracted)
- PET features: [SUVmax, SUVmean, SUVpeak, TLG, MTV]
- Clinical features: [Age, Sex, Smoking Status, Smoking Duration, Smoking Cessation, Tumor Diameter]
- Survival data: DFS time/event, OS time/event
- Classification label: LUAD (0) or LUSC (1)

Simulated data utilities are provided for experimentation and reproducibility.

---

## Installation

```bash
# Clone repository
conda create -n mm_dls python=3.10 -y
conda activate mm_dls
git clone https://github.com/your_username/MM-DLS-NSCLC.git
```
## Install dependencies
```bash
pip install -r requirements.txt
```
## Usage

### 🔽 Download Pretrained Models

Pretrained MM-DLS models are available for direct download:

- **MM-DLS (Full multimodal, best checkpoint)**  
  Download from: **[XX Download Link]([[https://xx.xx/mm-dls-pretrained-best.pt](https://drive.google.com/file/d/1IcyCwMgCX8wv0NMp84U4wlzhLoXH7ayx/view?usp=sharing)](https://drive.google.com/file/d/1IcyCwMgCX8wv0NMp84U4wlzhLoXH7ayx/view?usp=sharing))**

After downloading, place the model files under the `./MODEL/` directory:

Training:
```bash
python train_patient_model.py
```
Evaluation:
```bash
python test.py
```
Example Forward Pass:
```bash
python run_sample.ipynb
```
## Model Performance (from publication)
### Histological Subtype Classification:

AUC: 0.85 ~ 0.92 across cohorts

AP: 0.81 ~ 0.86

### TNM Stage Prediction:

AUC: Stage I-II (0.86 ~ 0.96), Stage III (0.85 ~ 0.95), Stage IV (0.83 ~ 0.95)

### AP and calibration maintained across internal and external sets

DFS & OS Prognosis:

C-index: up to 0.75

Time-dependent AUC (1/2/3 years): 0.77 ~ 0.91

Brier score: consistently < 0.2 for DFS and < 0.3 for OS

Superior to single modality models (clinical-only or imaging-only)

## Reference
Please cite our original publication when using this work:

License
This project is licensed under the MIT License.

Contact
For any questions or collaborations, please contact:

Dr. Fang Dai: daifang_cool@163.com
