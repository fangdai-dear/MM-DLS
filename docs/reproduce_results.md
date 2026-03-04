# Reproducing Results

This document describes how to reproduce the results reported in the paper.

---

## Step 1 Prepare dataset

Prepare the dataset following:

docs/dataset_format.md

---

## Step 2 Train model

Run:

python train_patient_model.py

---

## Step 3 Model outputs

After training, results will be stored in:

results/

including:

subtype predictions
TNM predictions
survival risk scores

---

## Step 4 Survival analysis

Kaplan–Meier curves and Cox regression statistics will be generated automatically.

Figures are saved in:

figures/

---

## Expected results

Typical performance:

Subtype classification AUC ≈ 0.88  
TNM prediction AUC ≈ 0.84  
DFS C-index ≈ 0.72  
OS C-index ≈ 0.75
