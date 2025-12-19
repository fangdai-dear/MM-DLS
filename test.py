# test_mm_dls.py
# =========================================================
# 🔍 Minimal test for MM-DLS pipeline
# - CUDA
# - forward / loss
# - pandas / lifelines (GLIBCXX check)
# =========================================================

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(".")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# Imports from mm_dls
# ---------------------------------------------------------
from mm_dls.HierMM_DLS import HierMM_DLS
from mm_dls.CoxphLoss import CoxPHLoss
from mm_dls.PatientDataset import PatientDataset


# =========================================================
# Basic config (VERY SMALL)
# =========================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

BATCH_SIZE = 2
NUM_SUBTYPES = 2
NUM_TNM = 3
N_SLICES = 30
IMG_SIZE = 224


# =========================================================
# Test Dataset Loader
# =========================================================
def get_test_loader():
    dataset = PatientDataset(
        data_root="/path/to/DATA_ROOT",
        clinical_csv="/path/to/clinical.csv",
        radiomics_npy="/path/to/radiomics.npy",
        pet_npy="/path/to/pet.npy",
        n_slices=N_SLICES,
        img_size=IMG_SIZE,
    )

    # 🔑 只取前 8 个样本
    idx = list(range(min(8, len(dataset))))
    subset = Subset(dataset, idx)

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )
    return loader


# =========================================================
# One forward + loss
# =========================================================
def test_forward_and_loss():
    print("\n[TEST] Forward + Loss")

    loader = get_test_loader()
    model = HierMM_DLS(NUM_SUBTYPES, NUM_TNM).to(DEVICE)

    ce  = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    cox = CoxPHLoss()

    model.eval()

    for batch in loader:
        assert len(batch) == 19, f"Dataset must return 19 items, got {len(batch)}"

        (
            pid, lesion, space, rad, pet, cli,
            y_sub, y_tnm,
            dfs_t, dfs_e,
            os_t, os_e,
            dfs1, dfs3, dfs5,
            os1, os3, os5,
            treatment
        ) = batch

        lesion, space = lesion.to(DEVICE), space.to(DEVICE)
        rad, pet, cli = rad.to(DEVICE), pet.to(DEVICE), cli.to(DEVICE)
        y_sub, y_tnm  = y_sub.to(DEVICE), y_tnm.to(DEVICE)
        dfs_t, dfs_e = dfs_t.to(DEVICE), dfs_e.to(DEVICE)
        os_t,  os_e  = os_t.to(DEVICE),  os_e.to(DEVICE)

        dfs_y = torch.stack([dfs1, dfs3, dfs5], dim=1).to(DEVICE)
        os_y  = torch.stack([os1,  os3,  os5 ], dim=1).to(DEVICE)

        with torch.no_grad():
            sub_l, tnm_l, dfs_r, os_r, dfs_log, os_log = model(
                lesion, space, rad, pet, cli
            )

            loss = (
                ce(sub_l, y_sub) +
                ce(tnm_l, y_tnm) +
                cox(dfs_r, dfs_t, dfs_e) +
                cox(os_r,  os_t,  os_e) +
                bce(dfs_log, dfs_y) +
                bce(os_log,  os_y)
            )

        print("  ✓ Forward OK | Loss =", float(loss))
        break


# =========================================================
# Test pandas + lifelines (GLIBCXX killer)
# =========================================================
def test_pandas_lifelines():
    print("\n[TEST] pandas + lifelines")

    # fake survival data
    time  = np.array([10, 12, 8, 20, 15, 25])
    event = np.array([1, 1, 0, 1, 0, 0])
    risk  = np.array([0.9, 0.8, 0.2, 1.2, 0.3, 0.4])

    # pandas
    df = pd.DataFrame({
        "time": time,
        "event": event,
        "risk": risk
    })

    print("  pandas OK:", df.shape)

    # C-index
    cidx = concordance_index(df["time"], -df["risk"], df["event"])
    print("  C-index =", round(cidx, 3))

    # KM
    kmf = KaplanMeierFitter()
    kmf.fit(df["time"], event_observed=df["event"])
    surv_10 = kmf.predict(10)

    print("  KM survival@10 =", float(surv_10))
    print("  ✓ lifelines OK")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print("\n==============================")
    print(" MM-DLS TEST START ")
    print("==============================")

    test_forward_and_loss()
    test_pandas_lifelines()

    print("\n✅ ALL TESTS PASSED")
