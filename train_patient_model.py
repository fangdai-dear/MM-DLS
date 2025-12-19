import sys
sys.path.insert(0, "/export/home/daifang/lunghospital/MM-DLS-master/MM-DLS-master")
# main.py
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.metrics import brier_score_loss
from scipy.stats import norm



# =========================================================
# Project path (IMPORTANT for Jupyter / HPC)
# =========================================================
PROJECT_ROOT = os.path.abspath(".")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================================================
# imports:  mm_dls/ 
# =========================================================
def _import_modules():

    from mm_dls.HierMM_DLS import HierMM_DLS
    from mm_dls.FakePatientDataset import FakePatientDataset
    from mm_dls.CoxphLoss import CoxPHLoss
    return HierMM_DLS, FakePatientDataset, CoxPHLoss


HierMM_DLS, FakePatientDataset, CoxPHLoss = _import_modules()


# =========================
# Training configuration
# =========================
EPOCHS        = 300
PATIENCE      = 8
BATCH_SIZE    = 4
LR            = 1e-4
WEIGHT_DECAY  = 1e-5

# =========================
# Task definition
# =========================
NUM_SUBTYPES  = 2        # e.g., LUAD vs LUSC
NUM_TNM       = 3        # Stage I–II / III / IV

# =========================
# Image settings
# =========================
N_SLICES      = 30       # max slices per patient
IMG_SIZE      = 224


SAVE_DIR = "./results"
FIG_DIR  = "./figures"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------------
# GPU (force cuda:1)
# -------------------------
assert torch.cuda.is_available(), "CUDA not available"
DEVICE = torch.device("cuda:1")
torch.cuda.set_device(DEVICE)
print("Using device:", DEVICE)


# =========================================================
# Core utils
# =========================================================
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _ensure_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def _risk_to_groups(risk, q=(1/3, 2/3), labels=("Low", "Mediate", "High")):
    """
    Convert continuous risk into 3 groups by tertiles.
    """
    r = np.asarray(risk).reshape(-1)
    t1, t2 = np.quantile(r, q[0]), np.quantile(r, q[1])
    out = np.full(len(r), labels[1], dtype=object)
    out[r <= t1] = labels[0]
    out[r >= t2] = labels[2]
    return out

def _evaluate_survival_metrics(time, event, risk, time_point=30):
    """
    C-index + Brier at a fixed time point.
    risk: higher => earlier event, so use -risk in concordance_index.
    """
    time = np.asarray(time).reshape(-1)
    event = np.asarray(event).reshape(-1).astype(int)
    risk = np.asarray(risk).reshape(-1)

    c_index = concordance_index(time, -risk, event)

    # Brier: predict survival at time_point using a monotonic transform of risk (proxy)
    # This is a "proxy" survival probability for demo/debug; replace with proper survival model if needed.
    y_true = (time > time_point).astype(int)  # 1 means survived beyond time_point
    # map risk into [0,1] survival prob proxy: higher risk => lower survival prob
    y_prob = 1 - (risk - risk.min()) / (risk.max() - risk.min() + 1e-8)
    brier = brier_score_loss(y_true, y_prob)

    return float(c_index), float(brier)


# =========================================================
# One epoch (train / eval)
# =========================================================
def run_epoch_verbose(model, loader, optimizer, device, train=True):
    ce  = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    cox = CoxPHLoss()

    model.train() if train else model.eval()

    losses = []

    # classification
    sub_y_all, sub_s_all = [], []
    tnm_y_all, tnm_s_all = [], []
    treat_all = []

    # survival (cox risk + time/event)
    dfs_r_all, dfs_t_all, dfs_e_all = [], [], []
    os_r_all,  os_t_all,  os_e_all  = [], [], []

    # survival 1y/3y/5y logits (optional save)
    dfs_log_all, os_log_all = [], []

    for batch in loader:
        # NOTE: dataset must return 19 items including treatment
        if len(batch) != 19:
            raise ValueError(f"Batch length mismatch: expected 19, got {len(batch)}. "
                             f"Please ensure Dataset __getitem__ returns treatment as the 19th item.")

        (
            pid, lesion, space, rad, pet, cli,
            y_sub, y_tnm,
            dfs_t, dfs_e,
            os_t, os_e,
            dfs1, dfs3, dfs5,
            os1, os3, os5,
            treatment
        ) = batch

        lesion, space = lesion.to(device), space.to(device)
        rad, pet, cli = rad.to(device), pet.to(device), cli.to(device)
        y_sub, y_tnm  = y_sub.to(device), y_tnm.to(device)
        dfs_t, dfs_e = dfs_t.to(device), dfs_e.to(device)
        os_t,  os_e  = os_t.to(device),  os_e.to(device)
        treatment = treatment.to(device)

        dfs_y = torch.stack([dfs1, dfs3, dfs5], dim=1).to(device)
        os_y  = torch.stack([os1,  os3,  os5 ], dim=1).to(device)

        with torch.set_grad_enabled(train):
            sub_l, tnm_l, dfs_r, os_r, dfs_log, os_log = model(
                lesion, space, rad, pet, cli
            )

            loss = (
                ce(sub_l, y_sub) +
                ce(tnm_l, y_tnm) +
                cox(dfs_r, dfs_t, dfs_e) +
                cox(os_r,  os_t,  os_e) +
                bce(dfs_log, dfs_y).mean() +
                bce(os_log,  os_y ).mean()
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(loss.item())

        # ----- Collect predictions -----
        sub_prob = torch.softmax(sub_l, dim=1)[:, 1]     # subtype prob
        tnm_prob = torch.softmax(tnm_l, dim=1)           # [B,3]

        sub_s_all.append(_ensure_numpy(sub_prob))
        sub_y_all.append(_ensure_numpy(y_sub))

        tnm_s_all.append(_ensure_numpy(tnm_prob))
        tnm_y_all.append(_ensure_numpy(y_tnm))

        treat_all.append(_ensure_numpy(treatment))

        # survival
        dfs_r_all.append(_ensure_numpy(dfs_r))
        dfs_t_all.append(_ensure_numpy(dfs_t))
        dfs_e_all.append(_ensure_numpy(dfs_e))

        os_r_all.append(_ensure_numpy(os_r))
        os_t_all.append(_ensure_numpy(os_t))
        os_e_all.append(_ensure_numpy(os_e))

        dfs_log_all.append(_ensure_numpy(dfs_log))
        os_log_all.append(_ensure_numpy(os_log))

    return (
        float(np.mean(losses)),

        np.concatenate(sub_y_all),
        np.concatenate(sub_s_all),

        np.concatenate(tnm_y_all),
        np.concatenate(tnm_s_all),

        np.concatenate(treat_all),

        np.concatenate(dfs_r_all),
        np.concatenate(dfs_t_all),
        np.concatenate(dfs_e_all),

        np.concatenate(os_r_all),
        np.concatenate(os_t_all),
        np.concatenate(os_e_all),

        np.concatenate(dfs_log_all, axis=0),  # [N,3]
        np.concatenate(os_log_all, axis=0),   # [N,3]
    )


# =========================================================
# Evaluation by cohort (classification + survival)
# =========================================================
def evaluate_by_treatment(sub_y, sub_s, tnm_y, tnm_s, treat,
                          dfs_r, dfs_t, dfs_e, os_r, os_t, os_e):
    results = {}

    cohorts = {
        "All": np.ones_like(treat, dtype=bool),
        "Immune": treat == 0,
        "Chemo":  treat == 1,
    }

    for name, mask in cohorts.items():
        if mask.sum() < 10:
            continue

        res = {}

        # Subtype (binary)
        res["Subtype_AUC"] = roc_auc_score(sub_y[mask], sub_s[mask])
        res["Subtype_ACC"] = accuracy_score(sub_y[mask], (sub_s[mask] > 0.5).astype(int))

        # TNM (multiclass macro AUC + ACC)
        tnm_bin = label_binarize(tnm_y[mask], classes=[0, 1, 2])
        res["TNM_AUC_macro"] = roc_auc_score(
            tnm_bin, tnm_s[mask], average="macro", multi_class="ovr"
        )
        res["TNM_ACC"] = accuracy_score(
            tnm_y[mask], np.argmax(tnm_s[mask], axis=1)
        )

        # Survival
        dfs_c, dfs_b = _evaluate_survival_metrics(dfs_t[mask], dfs_e[mask], dfs_r[mask], time_point=30)
        os_c,  os_b  = _evaluate_survival_metrics(os_t[mask],  os_e[mask],  os_r[mask],  time_point=30)

        res["DFS_C_index"] = dfs_c
        res["DFS_Brier_30m"] = dfs_b
        res["OS_C_index"] = os_c
        res["OS_Brier_30m"] = os_b

        results[name] = res

    return results


# =========================================================
# Figure 7: KM + HR (per cohort, per endpoint)
# =========================================================
def plot_km_curve_with_hr(df, title, save_prefix):
    """
    df must contain columns: time, event, group (Low/Mediate/High)
    """
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.set_facecolor("white")

    colors = {"Low": "#91c7ae", "Mediate": "#f7b977", "High": "#d87c7c"}
    groups = ["Low", "Mediate", "High"]

    # plot KM
    lines = {}
    at_risk_table = []
    times = np.arange(0, 70, 10)

    for g in groups:
        m = df["group"] == g
        if m.sum() == 0:
            continue

        kmf.fit(df.loc[m, "time"], event_observed=df.loc[m, "event"], label=g)
        kmf.plot_survival_function(
            ax=ax, ci_show=True, linewidth=2, color=colors[g], marker="+"
        )
        lines[g] = ax.get_lines()[-1]

        at_risk_table.append([np.sum(df.loc[m, "time"] >= t) for t in times])

    # legend
    handles = [lines[g] for g in groups if g in lines]
    labels = ["Low", "Medium", "High"][:len(handles)]
    ax.legend(handles, labels, title="Groups", loc="upper right",
              frameon=True, framealpha=0.5, fontsize=12, title_fontsize=12)

    # at risk numbers (optional, matches your style)
    if len(at_risk_table) == 3:
        low, mid, high = at_risk_table
        for i, t in enumerate(times):
            ax.text(t, -0.38, str(low[i]),  color="#207f4c", fontsize=14, ha="center")
            ax.text(t, -0.48, str(mid[i]),  color="#fca106", fontsize=14, ha="center")
            ax.text(t, -0.58, str(high[i]), color="#cc163a", fontsize=14, ha="center")

        ax.text(-1,  -0.28, "Number at risk", color="black", ha="center", fontsize=14)
        ax.text(-10, -0.38, "Low",    color="#207f4c", fontsize=14)
        ax.text(-10, -0.48, "Medium", color="#fca106", fontsize=14)
        ax.text(-10, -0.58, "High",   color="#cc163a", fontsize=14)

    # Cox HR + p-values
    df2 = df.copy()
    df2["group_code"] = df2["group"].map({"Low": 0, "Mediate": 1, "High": 2})
    cph = CoxPHFitter()
    cph.fit(df2[["time", "event", "group_code"]], duration_col="time", event_col="event")

    coef = float(cph.params_["group_code"])
    se   = float(cph.standard_errors_["group_code"])

    hr_med_vs_low  = np.exp(coef * 1)
    hr_high_vs_low = np.exp(coef * 2)

    z_med  = (coef * 1) / se
    p_med  = 2 * (1 - norm.cdf(abs(z_med)))

    z_high = (coef * 2) / se
    p_high = 2 * (1 - norm.cdf(abs(z_high)))

    # logrank
    res_lr = multivariate_logrank_test(df2["time"], df2["group"], df2["event"])

    # C-index + brier (proxy)
    c_index, brier = _evaluate_survival_metrics(df2["time"].values, df2["event"].values,
                                                df2["group_code"].values, time_point=30)

    ax.text(25, 0.46, f"P(log-rank)={res_lr.p_value:.3f}", fontsize=12)
    ax.text(25, 0.36, f"C-index={c_index:.3f}", fontsize=12)
    ax.text(25, 0.26, f"Brier(30m)={brier:.3f}", fontsize=12)
    ax.text(25, 0.16, f"HR Intermediate vs Low = {hr_med_vs_low:.2f}, P={p_med:.3f}", fontsize=12)
    ax.text(25, 0.06, f"HR High vs Low = {hr_high_vs_low:.2f}, P={p_high:.3f}", fontsize=12)

    # cosmetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time since treatment start (months)", fontsize=14)
    ax.set_ylabel("Survival probability", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_prefix + ".png", dpi=600, bbox_inches="tight")
    plt.savefig(save_prefix + ".pdf", dpi=600, bbox_inches="tight")
    plt.close()
    return save_prefix


def generate_figure_from_saved(result_dir=SAVE_DIR, fig_dir=FIG_DIR, which_split=("val", "test")):
    """
    Load saved dfs/os arrays and generate KM+HR for Immune/Chemo separately.
    """
    os.makedirs(fig_dir, exist_ok=True)

    for split in which_split:
        # load arrays
        trt = np.load(os.path.join(result_dir, f"treatment_{split}.npy"))

        dfs_r = np.load(os.path.join(result_dir, f"dfs_{split}_risk.npy"))
        dfs_t = np.load(os.path.join(result_dir, f"dfs_{split}_time.npy"))
        dfs_e = np.load(os.path.join(result_dir, f"dfs_{split}_event.npy"))

        os_r  = np.load(os.path.join(result_dir, f"os_{split}_risk.npy"))
        os_t  = np.load(os.path.join(result_dir, f"os_{split}_time.npy"))
        os_e  = np.load(os.path.join(result_dir, f"os_{split}_event.npy"))

        for cohort_name, mask in {
            "Immune": trt == 0,
            "Chemo":  trt == 1
        }.items():
            if mask.sum() < 20:
                print(f"[Figure7] Skip {split}-{cohort_name}: too few samples ({mask.sum()})")
                continue

            # DFS groups
            dfs_group = _risk_to_groups(dfs_r[mask])
            df_dfs = pd.DataFrame({
                "time": dfs_t[mask],
                "event": dfs_e[mask].astype(int),
                "group": dfs_group
            })

            # OS groups
            os_group = _risk_to_groups(os_r[mask])
            df_os = pd.DataFrame({
                "time": os_t[mask],
                "event": os_e[mask].astype(int),
                "group": os_group
            })

            # save CSV (optional, for reproducibility)
            df_dfs.to_csv(os.path.join(result_dir, f"dfs_{split}_{cohort_name}.csv"), index=False)
            df_os.to_csv(os.path.join(result_dir, f"os_{split}_{cohort_name}.csv"), index=False)

            # plot
            plot_km_curve_with_hr(
                df_dfs,
                title=f"Disease-Free Survival (DFS) — Kaplan-Meier Curves\n{cohort_name} {split} set (n={mask.sum()})",
                save_prefix=os.path.join(fig_dir, f"Figure7_DFS_{cohort_name}_{split}")
            )
            plot_km_curve_with_hr(
                df_os,
                title=f"Overall Survival (OS) — Kaplan-Meier Curves\n{cohort_name} {split} set (n={mask.sum()})",
                save_prefix=os.path.join(fig_dir, f"Figure7_OS_{cohort_name}_{split}")
            )

    print("✔ Figure 7 generated (DFS/OS KM + HR) for Immune/Chemo.")


# =========================================================
# Main
# =========================================================
def main():
    # -------------------------
    # Dataset (must return treatment as 19th item)
    # -------------------------
    from mm_dls.PatientDataset import PatientDataset

    dataset = PatientDataset(
        data_root="/path/to/DATA_ROOT",
        clinical_csv="/path/to/clinical.csv",
        radiomics_npy="/path/to/radiomics.npy",
        pet_npy="/path/to/pet.npy",
        n_slices=N_SLICES,
        img_size=IMG_SIZE,
    )


    n_train = int(0.6 * len(dataset))
    n_val   = int(0.2 * len(dataset))
    n_test  = len(dataset) - n_train - n_val

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    loaders = {
        "train": DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=4),
        "val":   DataLoader(val_set,   BATCH_SIZE, shuffle=False, num_workers=4),
        "test":  DataLoader(test_set,  BATCH_SIZE, shuffle=False, num_workers=4),
    }

    # -------------------------
    # Model
    # -------------------------
    model = HierMM_DLS(NUM_SUBTYPES, NUM_TNM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = 1e9
    wait = 0

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch_verbose(model, loaders["train"], optimizer, DEVICE, train=True)
        va = run_epoch_verbose(model, loaders["val"],   optimizer, DEVICE, train=False)

        tr_loss = tr[0]
        va_loss = va[0]

        # unpack val for metrics
        _, sy, ss, ty, ts, trt, dfs_r, dfs_t, dfs_e, os_r, os_t, os_e, _, _ = va
        metrics = evaluate_by_treatment(sy, ss, ty, ts, trt, dfs_r, dfs_t, dfs_e, os_r, os_t, os_e)

        print(f"\n[Epoch {epoch:03d}] Train Loss={tr_loss:.3f} | Val Loss={va_loss:.3f}")
        for k, v in metrics.items():
            print(
                f"  {k:7s} | "
                f"Subtype AUC={v['Subtype_AUC']:.3f} | "
                f"TNM AUC={v['TNM_AUC_macro']:.3f} | "
                f"DFS C-index={v['DFS_C_index']:.3f} | "
                f"OS C-index={v['OS_C_index']:.3f}"
            )

        # early stopping
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            wait = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
            print("  ✓ Best model updated")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("\n⏹ Early stopping triggered")
                break

    # -------------------------
    # Inference (best model)
    # -------------------------
    print("\nRunning inference with best model...")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pt"), map_location=DEVICE))

    for split in ["train", "val", "test"]:
        out = run_epoch_verbose(model, loaders[split], optimizer, DEVICE, train=False)
        (
            loss,
            sy, ss,
            ty, ts,
            trt,
            dfs_r, dfs_t, dfs_e,
            os_r,  os_t,  os_e,
            dfs_log, os_log
        ) = out

        # classification
        np.save(os.path.join(SAVE_DIR, f"subtype_{split}_labels.npy"), sy)
        np.save(os.path.join(SAVE_DIR, f"subtype_{split}_scores.npy"), ss)
        np.save(os.path.join(SAVE_DIR, f"tnm_{split}_labels.npy"), ty)
        np.save(os.path.join(SAVE_DIR, f"tnm_{split}_scores.npy"), ts)
        np.save(os.path.join(SAVE_DIR, f"treatment_{split}.npy"), trt)

        # survival (cox risk + time/event)
        np.save(os.path.join(SAVE_DIR, f"dfs_{split}_risk.npy"),  dfs_r)
        np.save(os.path.join(SAVE_DIR, f"dfs_{split}_time.npy"),  dfs_t)
        np.save(os.path.join(SAVE_DIR, f"dfs_{split}_event.npy"), dfs_e)

        np.save(os.path.join(SAVE_DIR, f"os_{split}_risk.npy"),   os_r)
        np.save(os.path.join(SAVE_DIR, f"os_{split}_time.npy"),   os_t)
        np.save(os.path.join(SAVE_DIR, f"os_{split}_event.npy"),  os_e)

        # 1y/3y/5y logits (optional, for AUC at specific horizons)
        np.save(os.path.join(SAVE_DIR, f"dfs_{split}_logits_1y3y5y.npy"), dfs_log)
        np.save(os.path.join(SAVE_DIR, f"os_{split}_logits_1y3y5y.npy"),  os_log)

        print(f"{split:5s} | loss={loss:.3f} | Immune={np.sum(trt==0)} Chemo={np.sum(trt==1)}")

    print("\n✓ Inference completed. Results saved.")

    # -------------------------
    # Figure: Immune/Chemo KM + HR
    # -------------------------
    print("\nGenerating Figure  (KM + HR) ...")
    generate_figure_from_saved(result_dir=SAVE_DIR, fig_dir=FIG_DIR, which_split=("val", "test"))
    print("✓ Figure  done. Files saved under ./figures")


if __name__ == "__main__":
    main()

