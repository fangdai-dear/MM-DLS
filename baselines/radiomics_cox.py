#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radiomics + Cox baseline (patient-level).
- Input: clinical.csv + radiomics features (.npy) or .csv
- Output: risk scores + metrics (C-index; optional time-dependent AUC if available)
- Designed for reproducibility and reviewer-friendly transparency.

Example:
python baselines/radiomics_cox.py \
  --clinical_csv data/clinical.csv \
  --radiomics_path data/radiomics.npy \
  --endpoint DFS \
  --id_col patient_id \
  --time_col dfs_time \
  --event_col dfs_event \
  --output_dir results/radiomics_cox
"""

import os
import json
import argparse
import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_radiomics(radiomics_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      X: (N, D)
      ids: (N,) or None
    Supports:
      - .npy with array only (N,D)
      - .npz with keys: X and optionally ids
      - .csv where first col can be patient_id or any id column; use --radiomics_id_col
    """
    if radiomics_path.lower().endswith(".npy"):
        X = np.load(radiomics_path)
        return X, None
    if radiomics_path.lower().endswith(".npz"):
        z = np.load(radiomics_path, allow_pickle=True)
        if "X" not in z:
            raise ValueError("NPZ must contain key 'X'.")
        X = z["X"]
        ids = z["ids"] if "ids" in z else None
        return X, ids
    if radiomics_path.lower().endswith(".csv"):
        df = pd.read_csv(radiomics_path)
        return df.values, None
    raise ValueError(f"Unsupported radiomics_path: {radiomics_path}")


def zscore_fit_transform(X_train: np.ndarray, X_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    return (X_train - mu) / sd, (X_other - mu) / sd, mu, sd


def variance_filter(X: np.ndarray, thr: float = 1e-8) -> np.ndarray:
    v = X.var(axis=0)
    keep = v > thr
    return keep


def correlation_filter(X: np.ndarray, thr: float = 0.95) -> np.ndarray:
    """
    Greedy correlation filter to reduce multicollinearity.
    Returns boolean mask of kept features.
    """
    if X.shape[1] <= 1:
        return np.ones((X.shape[1],), dtype=bool)

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    keep = np.ones((X.shape[1],), dtype=bool)
    for i in range(X.shape[1]):
        if not keep[i]:
            continue
        # remove features j>i highly correlated with i
        for j in range(i + 1, X.shape[1]):
            if keep[j] and abs(corr[i, j]) >= thr:
                keep[j] = False
    return keep


def try_time_dependent_auc(times: np.ndarray, events: np.ndarray, risks: np.ndarray, eval_times: list):
    """
    Optional: compute time-dependent AUC.
    This requires scikit-survival (sksurv). If not installed, we skip gracefully.
    """
    try:
        from sksurv.metrics import cumulative_dynamic_auc
        from sksurv.util import Surv
    except Exception:
        return None

    y = Surv.from_arrays(events.astype(bool), times.astype(float))
    # cumulative_dynamic_auc needs train/test; for a simple baseline we compute on same set
    # (you can change to use an external test set if needed).
    aucs, mean_auc = cumulative_dynamic_auc(y, y, risks, np.array(eval_times, dtype=float))
    out = {str(t): float(a) for t, a in zip(eval_times, aucs)}
    out["mean_auc"] = float(mean_auc)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical_csv", required=True, help="Path to clinical CSV.")
    parser.add_argument("--radiomics_path", required=True, help="Path to radiomics features (.npy/.npz/.csv).")
    parser.add_argument("--endpoint", default="DFS", choices=["DFS", "OS"], help="Survival endpoint label.")
    parser.add_argument("--id_col", default="patient_id", help="Patient ID column in clinical CSV.")
    parser.add_argument("--time_col", default=None, help="Survival time column. If None, uses dfs_time/os_time by endpoint.")
    parser.add_argument("--event_col", default=None, help="Event indicator column. If None, uses dfs_event/os_event by endpoint.")
    parser.add_argument("--split_col", default=None, help="Optional split column name in clinical CSV (e.g., split: train/val/test).")
    parser.add_argument("--train_split_value", default="train", help="Value in split_col for training.")
    parser.add_argument("--test_split_value", default="test", help="Value in split_col for testing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corr_thr", type=float, default=0.95)
    parser.add_argument("--var_thr", type=float, default=1e-8)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_times", type=str, default="1,3,5", help="Comma-separated years for AUC (optional).")
    parser.add_argument("--time_unit_in_years", action="store_true",
                        help="If set, time is in years already; otherwise we assume days and convert years->days for eval_times.")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    clin = pd.read_csv(args.clinical_csv)
    if args.time_col is None:
        args.time_col = "dfs_time" if args.endpoint == "DFS" else "os_time"
    if args.event_col is None:
        args.event_col = "dfs_event" if args.endpoint == "DFS" else "os_event"

    required_cols = [args.id_col, args.time_col, args.event_col]
    for c in required_cols:
        if c not in clin.columns:
            raise ValueError(f"Missing column in clinical_csv: {c}")

    # Load radiomics
    X, ids = load_radiomics(args.radiomics_path)

    # Align rows by patient_id if possible
    # Option A: radiomics is .npz with ids
    if ids is not None:
        rad_df = pd.DataFrame(X)
        rad_df[args.id_col] = ids
        merged = clin.merge(rad_df, on=args.id_col, how="inner")
        if merged.shape[0] == 0:
            raise ValueError("No matching patient IDs between clinical_csv and radiomics ids.")
        clin = merged
        feat_cols = [c for c in clin.columns if c not in [args.id_col, args.time_col, args.event_col, args.split_col]]
        X_all = clin[feat_cols].values.astype(np.float32)
    else:
        # Option B: assume same order (warn user in outputs)
        if X.shape[0] != clin.shape[0]:
            raise ValueError(
                f"Radiomics rows ({X.shape[0]}) != clinical rows ({clin.shape[0]}). "
                "Provide .npz with 'ids' to align, or ensure same order and same N."
            )
        X_all = X.astype(np.float32)
        feat_cols = [f"rad_{i}" for i in range(X_all.shape[1])]

    times = clin[args.time_col].values.astype(float)
    events = clin[args.event_col].values.astype(int)

    # Split handling
    if args.split_col and args.split_col in clin.columns:
        train_mask = clin[args.split_col].astype(str).str.lower() == str(args.train_split_value).lower()
        test_mask = clin[args.split_col].astype(str).str.lower() == str(args.test_split_value).lower()
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            raise ValueError("Split masks are empty. Check split_col/train_split_value/test_split_value.")
    else:
        # If no split info, use all as one set (still reproducible)
        train_mask = np.ones((clin.shape[0],), dtype=bool)
        test_mask = np.ones((clin.shape[0],), dtype=bool)

    X_train = X_all[train_mask]
    X_test = X_all[test_mask]
    t_train = times[train_mask]
    e_train = events[train_mask]
    t_test = times[test_mask]
    e_test = events[test_mask]

    # Preprocess
    keep_var = variance_filter(X_train, thr=args.var_thr)
    X_train = X_train[:, keep_var]
    X_test = X_test[:, keep_var]
    kept_feat_names = np.array(feat_cols)[keep_var].tolist()

    X_train, X_test, mu, sd = zscore_fit_transform(X_train, X_test)
    keep_corr = correlation_filter(X_train, thr=args.corr_thr)
    X_train = X_train[:, keep_corr]
    X_test = X_test[:, keep_corr]
    kept_feat_names = np.array(kept_feat_names)[keep_corr].tolist()

    # Build training dataframe for lifelines
    df_train = pd.DataFrame(X_train, columns=kept_feat_names)
    df_train["time"] = t_train
    df_train["event"] = e_train

    cph = CoxPHFitter(penalizer=0.1)  # mild L2 regularization for stability
    cph.fit(df_train, duration_col="time", event_col="event")

    # Predict risk scores (partial hazard)
    df_test_x = pd.DataFrame(X_test, columns=kept_feat_names)
    risk_test = cph.predict_partial_hazard(df_test_x).values.astype(float)
    risk_train = cph.predict_partial_hazard(pd.DataFrame(X_train, columns=kept_feat_names)).values.astype(float)

    # C-index (higher risk => shorter survival, so use -risk or risk depending convention)
    # lifelines concordance_index assumes higher score => longer survival unless event is handled.
    # We use -risk so that higher score => longer survival.
    cindex_train = concordance_index(t_train, -risk_train, e_train)
    cindex_test = concordance_index(t_test, -risk_test, e_test)

    # Optional time-dependent AUC
    eval_years = [float(x) for x in args.eval_times.split(",") if x.strip() != ""]
    eval_times = eval_years if args.time_unit_in_years else [y * 365.0 for y in eval_years]
    auc_info = try_time_dependent_auc(t_test, e_test, risk_test, eval_times)

    # Save outputs
    out_pred = pd.DataFrame({
        args.id_col: clin.loc[test_mask, args.id_col].values,
        "time": t_test,
        "event": e_test,
        "risk": risk_test
    })
    out_pred.to_csv(os.path.join(args.output_dir, f"{args.endpoint.lower()}_predictions.csv"), index=False)

    summary = {
        "endpoint": args.endpoint,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "cindex_train": float(cindex_train),
        "cindex_test": float(cindex_test),
        "time_dependent_auc": auc_info,
        "preprocess": {
            "variance_threshold": args.var_thr,
            "corr_threshold": args.corr_thr,
            "zscore": True,
            "penalizer": 0.1
        },
        "alignment": "id-aligned" if ids is not None else "assumed-same-order",
        "kept_features": int(len(kept_feat_names))
    }
    with open(os.path.join(args.output_dir, f"{args.endpoint.lower()}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[Radiomics+Cox] Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
