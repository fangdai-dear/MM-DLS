# 🧠 MM-DLS: Multimodal Deep Learning Survival Model

Implementation of the multimodal deep learning framework described in the study:

**Assessment of Bone Metastasis via Deep Learning Applied to CT Images**

This repository contains the full training, evaluation, and analysis pipeline for the **MM-DLS (Multimodal Deep Learning Survival)** framework.

The MM-DLS framework integrates **multimodal imaging features, radiomics features, PET features, and clinical variables** to jointly perform:

- Tumor subtype classification
- TNM stage prediction
- Survival risk prediction
- Kaplan–Meier survival analysis
- Cox proportional hazards modeling
- Risk stratification for clinical cohorts

---

# 📑 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Baseline Models](#Baseline Models)
- [Running Evaluation](#running-evaluation)
- [Minimal Test](#minimal-test)
- [Reproducing Results](#reproducing-results)
- [Code Availability](#code-availability)
- [License](#license)
- [Citation](#citation)

---

# 🔬 Overview

The **MM-DLS** framework is designed for multimodal medical imaging analysis and survival prediction.

The system integrates multiple sources of information:

| Data Type | Description |
|-----------|-------------|
| CT imaging | Tumor lesion imaging slices |
| Spatial imaging features | Structural spatial features |
| Radiomics | Quantitative radiomics descriptors |
| PET imaging | Metabolic imaging features |
| Clinical variables | Patient demographic and clinical data |

The model predicts:

- **Tumor subtype classification**
- **TNM stage classification**
- **Disease-Free Survival (DFS)**
- **Overall Survival (OS)**

The framework further performs:

- Kaplan–Meier survival analysis
- Cox proportional hazards modeling
- Risk stratification into **Low / Intermediate / High** risk groups

---

# 📂 Repository Structure

```
MM-DLS/
│
├── mm_dls/                    # Core model architecture and dataset classes
│
├── MODELS/                    # Pretrained or auxiliary model components
│
├── train_patient_model.py     # Main training script
├── test.py                    # Minimal pipeline test
├── run_sample.ipynb           # Example notebook for running inference
│
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
│
└── docs/
    ├── dataset_format.md
    └── reproduce_results.md
```

---

# ⚙️ Installation

Create a Python environment:

```bash
conda create -n mmdls python=3.9
conda activate mmdls
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

# 📦 Dependencies

Main packages used in this project:

| Package | Purpose |
|--------|--------|
| PyTorch | Deep learning framework |
| NumPy | Numerical computing |
| Pandas | Data processing |
| scikit-learn | Evaluation metrics |
| lifelines | Survival analysis |
| matplotlib | Visualization |
| scipy | Scientific computation |

Full dependency list is provided in:

```
requirements.txt
```

---

# 🗂 Dataset Preparation

The dataset includes multimodal patient data:

- CT image slices
- Spatial imaging features
- Radiomics features
- PET imaging features
- Clinical variables
- Survival outcomes

Detailed dataset structure is described in:

```
docs/dataset_format.md
```

Example dataset structure:

```
DATA_ROOT/

    patient001/
        lesion/
            slice_01.png
            slice_02.png
        space/
            slice_01.png
            slice_02.png

    patient002/
        lesion/
        space/

clinical.csv
radiomics.npy
pet.npy
```

### Clinical Variables

`clinical.csv` contains:

- patient_id
- age
- sex
- treatment type
- DFS time
- DFS event
- OS time
- OS event

---

⚠️ **Data Availability**

Due to **patient privacy and institutional ethical regulations**, the clinical datasets used in this study cannot be publicly released.

---

# 🧠 Training the Model

To train the model:

```bash
python train_patient_model.py
```

The training script performs:

- data loading
- multimodal feature integration
- model training
- validation
- early stopping
- model checkpoint saving

Training outputs will be saved in:

```
results/
```

---

# 📊 Running Evaluation

After training, inference will automatically generate:

- classification predictions
- survival risk scores
- Kaplan–Meier survival curves
- hazard ratio estimates

Generated outputs will be stored in:

```
results/
figures/
```

Example generated outputs:

| Output | Description |
|------|-------------|
| subtype_scores.npy | Subtype classification predictions |
| tnm_scores.npy | TNM stage predictions |
| dfs_risk.npy | DFS risk scores |
| os_risk.npy | OS risk scores |
| KM_curves.png | Kaplan–Meier survival plots |

---

# 🧪 Minimal Test

To verify that the environment and model pipeline are correctly configured:

```bash
python test.py
```

This script tests:

- CUDA availability
- model forward pass
- loss computation
- survival analysis utilities
- pandas / lifelines compatibility

If the environment is correctly configured, the script will print:

```
ALL TESTS PASSED
```

---

# 🔁 Reproducing Results

Detailed instructions for reproducing the results reported in the manuscript are provided in:

```
docs/reproduce_results.md
```

General workflow:

### Step 1

Prepare dataset following

```
docs/dataset_format.md
```

### Step 2

Train the model

```bash
python train_patient_model.py
```

### Step 3

Generate predictions and survival analysis

Outputs will appear in:

```
results/
figures/
```

Expected model performance:

| Metric | Expected Value |
|------|------|
| Subtype classification AUC | ~0.88 |
| TNM prediction AUC | ~0.84 |
| DFS C-index | ~0.72 |
| OS C-index | ~0.75 |

---

## Baseline Models

To ensure a fair and reproducible comparison, several representative baseline models were re-implemented and evaluated on the same dataset splits as the proposed MM-DLS framework. All baseline models were trained and evaluated using identical training/validation/test partitions to avoid dataset bias.

The baseline implementations are provided in the `baselines/` directory.

```
baselines/
├── radiomics_cox.py
└── ct_cnn_survival.py
```

These scripts reproduce two commonly used methodological paradigms in medical imaging prognosis studies:

1. Radiomics-based survival prediction models  
2. Single-modality deep learning survival prediction models  

Both baselines are trained using the same input cohort used for the MM-DLS model and are evaluated using the same survival metrics (C-index and time-dependent AUC).

---

# Radiomics + Cox Survival Model

File:

```
baselines/radiomics_cox.py
```

This baseline implements a classical radiomics-based survival prediction pipeline, which is widely used in oncologic imaging studies.

### Pipeline Overview

The workflow consists of four steps:

1. Tumor segmentation  
2. Radiomics feature extraction  
3. Feature normalization  
4. Cox proportional hazards modeling

### Feature Extraction

Radiomics features are extracted from CT images using **PyRadiomics**, including:

First-order features
- Mean intensity
- Standard deviation
- Skewness
- Kurtosis

Texture features

Gray Level Co-occurrence Matrix (GLCM)

Gray Level Run Length Matrix (GLRLM)

Gray Level Size Zone Matrix (GLSZM)

Neighbouring Gray Tone Difference Matrix (NGTDM)

A total of approximately **100–120 radiomics features** are generated per lesion.

### Feature Processing

The following preprocessing steps are applied:

- Z-score normalization  
- Removal of near-zero variance features  
- Optional correlation filtering  

### Survival Model

The processed radiomics features are used to train a **Cox proportional hazards model**.

The Cox model estimates patient-level risk scores for:

Disease-Free Survival (DFS)

Overall Survival (OS)

### Evaluation Metrics

Performance is evaluated using:

Concordance Index (C-index)

Time-dependent ROC AUC (1-year, 3-year, and 5-year survival)

These metrics match those reported in the manuscript.

### Running the Baseline

Example command:

```bash
python baselines/radiomics_cox.py \
    --clinical_csv data/clinical.csv \
    --radiomics_features data/radiomics.npy \
    --output_dir results/radiomics
```

The script outputs:

```
results/
├── dfs_predictions.csv
├── os_predictions.csv
├── cindex_results.json
```

---

# CT-CNN Survival Model

File:

```
baselines/ct_cnn_survival.py
```

This baseline implements a **single-modality deep learning survival prediction model using CT images only**, which represents a commonly used architecture in imaging-based prognosis studies.

### Model Architecture

The model contains three components:

CT Feature Encoder

A convolutional neural network (CNN) extracts imaging features from CT slices.

Example backbone:

ResNet-18 / ResNet-34

Feature Aggregation

Slice-level features are aggregated into a patient-level representation using:

Average pooling  
or  
Attention pooling

Survival Prediction Head

The aggregated feature vector is fed into a survival prediction module:

Fully connected layers

Cox proportional hazards regression layer

### Training Objective

The model is optimized using **Cox partial likelihood loss**, defined as:

L = − Σ_i (h_i − log Σ_j exp(h_j))

where:

h_i is the predicted risk score.

### Input Data

The model uses only CT imaging data.

Inputs:

```
CT images
Tumor segmentation masks
Survival time
Event indicator
```

No PET or clinical variables are used.

### Evaluation

The model predicts patient-level risk scores, which are evaluated using:

C-index

Kaplan–Meier survival stratification

Time-dependent ROC curves

### Running the Baseline

Example command:

```bash
python baselines/ct_cnn_survival.py \
    --data_root data/ct_images \
    --clinical_csv data/clinical.csv \
    --output_dir results/ct_cnn
```

Output files include:

```
results/
├── dfs_predictions.csv
├── os_predictions.csv
├── cindex_results.json
```


# Comparison with the Proposed Model

All baseline models were evaluated using the same dataset partitions and evaluation metrics as MM-DLS.

The proposed **MM-DLS** model differs from the baselines in several key aspects:

Multimodal fusion  
The proposed model integrates CT imaging, PET metabolic features, and clinical variables.

Hierarchical multi-task learning  
MM-DLS simultaneously predicts histologic subtype, TNM stage, and survival risk.

Cross-modal attention fusion  
The architecture models interactions between imaging modalities and clinical information.

These design choices enable MM-DLS to achieve improved prognostic prediction performance compared with single-modality radiomics or CNN-based baselines.

---

# Reproducibility

All baseline experiments were conducted using the same training pipeline, dataset splits, and evaluation protocols as the MM-DLS model to ensure fair comparison and reproducibility.

```

---

# 💻 Code Availability

The source code used for model development and analysis is publicly available at:

```
https://github.com/XXXX/MM-DLS
```

A citable archived version of the repository is available via Zenodo:

```
DOI: 10.5281/zenodo.xxxxxx
```

---

# 📜 License

This project is released under the **MIT License**.

See:

```
LICENSE
```

for details.

---

# 📖 Citation

If you use this code in your research, please cite:

```
Dai F. et al.
MM-DLS: Multimodal Deep Learning Survival Model.

GitHub repository.
DOI: 10.5281/zenodo.xxxxxx
```

Machine-readable citation metadata is provided in:

```
CITATION.cff
```

---

# 🤝 Acknowledgements

This work integrates multimodal medical imaging and clinical data to enable robust survival prediction and risk stratification for oncology applications.

We thank all collaborators and clinical contributors involved in data collection and annotation.

---

# ⭐ If this repository helps your research

Please consider **starring the repository** on GitHub.
