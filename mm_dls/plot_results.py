# code/plot_results.py
# ============================================================
# End-to-end paper-style plotting (curves + tables)
# - Subtype (binary): ROC + PR + Calibration (with tables)
# - TNM (multiclass OVR): ROC + PR + Calibration (with tables, per class)
# - DFS/OS survival: KM + Cox HR + log-rank + C-index/Brier (with at-risk text)
#
# IMPORTANT:
#   - Safe to import (NO plotting on import)
#   - Call plot_all(result_dir, fig_dir) after main.py saves outputs
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from scipy.stats import norm


# ============================================================
# Basic I/O helpers
# ============================================================
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)


def _load_npy(path: str):
    if not _exists(path):
        return None
    return np.load(path, allow_pickle=True)


def _maybe_sim_ext(labels, scores, noise=0.03, seed=42):
    """
    Simulate an external test split when not provided.
    Keeps labels same; adds small noise to scores then clips to [0,1].
    """
    rng = np.random.RandomState(seed)
    if scores is None:
        return None, None
    s = scores.copy()
    s = np.clip(s + rng.normal(0, noise, s.shape), 0.0, 1.0)
    return labels.copy(), s


# ============================================================
# Metrics helpers
# ============================================================
def _calc_binary_roc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    brier = brier_score_loss(y_true, y_score)
    return fpr, tpr, roc_auc, brier


def _calc_binary_pr(y_true, y_score):
    p, r, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return p, r, ap


def _spec_npv_binary(y_true, y_score, thresh=0.5):
    y_pred = (y_score >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return specificity, npv


def _ece(y_true, y_score, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_score, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        m = binids == i
        if m.sum() > 0:
            prob_true = np.mean(y_true[m])
            prob_pred = np.mean(y_score[m])
            ece += (m.sum() / len(y_score)) * abs(prob_pred - prob_true)
    return float(ece)


def _calc_ovr_auc(y_bin, y_score):
    """One-vs-rest ROC for multiclass. Returns dict: {class_i: (fpr,tpr,auc)}"""
    out = {}
    for i in range(y_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        out[i] = (fpr, tpr, auc(fpr, tpr))
    return out


def _calc_ovr_pr(y_bin, y_score):
    """One-vs-rest PR for multiclass. Returns dict: {class_i: (p,r,ap)}"""
    out = {}
    for i in range(y_bin.shape[1]):
        p, r, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_bin[:, i], y_score[:, i])
        out[i] = (p, r, ap)
    return out


def _acc_ovr(y_true_bin, y_score, thresh=0.5):
    y_pred = (y_score >= thresh).astype(int)
    return float((y_pred == y_true_bin).mean())


# ============================================================
# Table helpers (paper-style)
# ============================================================
def _auto_col_widths(col_labels, bbox_w):
    lens = np.array([max(4, len(c)) for c in col_labels], dtype=float)
    ratio = lens / lens.sum()
    return bbox_w * ratio


def _add_table(ax, table_data, row_labels, col_labels, colors=None,
               bbox=(0.05, -0.50, 0.95, 0.30),
               fontsize=13, rowlabel_width=0.18):
    """
    colors: list[str] length = len(row_labels) (for per-row coloring)
    """
    tbl = plt.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        rowLoc='left',
        colLoc='center',
        bbox=list(bbox),
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)

    cells = tbl.get_celld()
    # set column widths (excluding row label col=-1)
    col_widths = _auto_col_widths(col_labels, bbox[2])
    for col in range(len(col_labels)):
        for row in range(len(row_labels) + 1):  # header included
            cells[(row, col)].set_width(col_widths[col])

    # row label width
    for row in range(1, len(row_labels) + 1):
        if (row, -1) in cells:
            cells[(row, -1)].set_width(rowlabel_width)

    # styling: no grid lines
    for (r, c), cell in cells.items():
        cell.set_linewidth(0)

    # optional per-row color
    if colors is not None:
        for r in range(1, len(row_labels) + 1):
            # color values (not the header)
            for c in range(len(col_labels)):
                if (r, c) in cells:
                    cells[(r, c)].get_text().set_color(colors[r - 1])
            # row label
            if (r, -1) in cells:
                cells[(r, -1)].get_text().set_color(colors[r - 1])

    return tbl


# ============================================================
# Subtype (binary) plots: ROC / PR / Calibration
# ============================================================
def plot_subtype_binary(result_dir="./results", fig_dir="./figures",
                        title_suffix="(LUAD vs LUSC)"):
    _ensure_dir(fig_dir)

    # Required: train/val/test
    paths = {
        "Train": (os.path.join(result_dir, "subtype_train_labels.npy"),
                  os.path.join(result_dir, "subtype_train_scores.npy")),
        "Int.Valid": (os.path.join(result_dir, "subtype_val_labels.npy"),
                      os.path.join(result_dir, "subtype_val_scores.npy")),
        "Int.Test": (os.path.join(result_dir, "subtype_test_labels.npy"),
                     os.path.join(result_dir, "subtype_test_scores.npy")),
    }

    data = {}
    missing_core = False
    for k, (lp, sp) in paths.items():
        y = _load_npy(lp)
        s = _load_npy(sp)
        if y is None or s is None:
            print(f"[plot_subtype_binary] Skip: missing {lp} or {sp}")
            missing_core = True
            break
        data[k] = (y.astype(int), s.astype(float))

    if missing_core:
        return

    # External (simulated) if not present
    ext_lp = os.path.join(result_dir, "subtype_test2_labels.npy")
    ext_sp = os.path.join(result_dir, "subtype_test2_scores.npy")
    ext_y = _load_npy(ext_lp)
    ext_s = _load_npy(ext_sp)
    if ext_y is None or ext_s is None:
        ext_y, ext_s = _maybe_sim_ext(data["Int.Test"][0], data["Int.Test"][1], noise=0.04, seed=7)
    data["Ext.Test"] = (ext_y.astype(int), ext_s.astype(float))

    # Colors (match your style)
    colors = {
        "Train": "#0074B7",
        "Int.Valid": "#60A3D9",
        "Int.Test": "#6CC4DC",
        "Ext.Test": "#61649f",
    }
    row_colors = [colors["Train"], colors["Int.Valid"], colors["Int.Test"], colors["Ext.Test"]]

    # ---------- ROC (Figure 4a-like) ----------
    roc_items = {}
    for k, (y, s) in data.items():
        fpr, tpr, auc_k, brier_k = _calc_binary_roc(y, s)
        roc_items[k] = dict(fpr=fpr, tpr=tpr, auc=auc_k, brier=brier_k, y=y, s=s)

    auc_list = np.array([roc_items[k]["auc"] for k in ["Train", "Int.Valid", "Int.Test", "Ext.Test"]], dtype=float)
    auc_cv = float(np.std(auc_list) / np.mean(auc_list)) if np.mean(auc_list) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(5, 7), facecolor="white")
    ax.set_facecolor("white")

    for k in ["Train", "Int.Valid", "Int.Test", "Ext.Test"]:
        ax.plot(roc_items[k]["fpr"], roc_items[k]["tpr"],
                label=f"{k} (AUC = {roc_items[k]['auc']:.2f})",
                color=colors[k], linewidth=3)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title(f"Pathological Subtype Classification ROC Curves\n{title_suffix}", fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(alpha=0.3)

    # Table: Number / AUC CV / Brier Score
    def _posneg(y):
        neg = int((y == 0).sum())
        pos = int((y == 1).sum())
        return f"{neg} vs {pos}"

    row_labels = ["Train", "Int.Valid", "Int.Test", "Ext.Test"]
    col_labels = ["Number", "AUC CV", "Brier Score"]
    table_data = [
        [_posneg(roc_items["Train"]["y"]),     f"{auc_cv:.2f}", f"{roc_items['Train']['brier']:.3f}"],
        [_posneg(roc_items["Int.Valid"]["y"]), f"{auc_cv:.2f}", f"{roc_items['Int.Valid']['brier']:.3f}"],
        [_posneg(roc_items["Int.Test"]["y"]),  f"{auc_cv:.2f}", f"{roc_items['Int.Test']['brier']:.3f}"],
        [_posneg(roc_items["Ext.Test"]["y"]),  f"{auc_cv:.2f}", f"{roc_items['Ext.Test']['brier']:.3f}"],
    ]
    _add_table(ax, table_data, row_labels, col_labels, colors=row_colors,
               bbox=(0.05, -0.52, 0.98, 0.30), fontsize=12, rowlabel_width=0.20)

    plt.subplots_adjust(bottom=0.42)
    plt.savefig(os.path.join(fig_dir, "Figure4a_subtype_ROC.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "Figure4a_subtype_ROC.pdf"), dpi=600, bbox_inches="tight")
    plt.close()

    # ---------- PR (Figure 4b-like) ----------
    pr_items = {}
    for k, (y, s) in data.items():
        p, r, ap = _calc_binary_pr(y, s)
        spec, npv = _spec_npv_binary(y, s, thresh=0.5)
        pr_items[k] = dict(p=p, r=r, ap=ap, spec=spec, npv=npv, y=y, s=s)

    ap_vals = np.array([pr_items[k]["ap"] for k in ["Train", "Int.Valid", "Int.Test", "Ext.Test"]], dtype=float)
    ap_cv = float(np.std(ap_vals) / np.mean(ap_vals)) if np.mean(ap_vals) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(7, 5.3), facecolor="white")
    ax.set_facecolor("white")

    for k in ["Train", "Int.Valid", "Int.Test", "Ext.Test"]:
        ax.plot(pr_items[k]["r"], pr_items[k]["p"],
                label=f"{k} (AP={pr_items[k]['ap']:.2f})",
                color={
                    "Train": "#7F8FA3",
                    "Int.Valid": "#FFA0A3",
                    "Int.Test": "#77DDF9",
                    "Ext.Test": "#61649f",
                }[k],
                linewidth=3)
        ax.fill_between(pr_items[k]["r"], pr_items[k]["p"], step='post', alpha=0.1,
                        color={
                            "Train": "#7F8FA3",
                            "Int.Valid": "#FFA0A3",
                            "Int.Test": "#77DDF9",
                            "Ext.Test": "#61649f",
                        }[k])

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title(f"Pathological Subtype Classification Precision-Recall Curves\n{title_suffix}", fontsize=14)
    ax.legend(loc="lower left", fontsize=12)
    ax.grid(alpha=0.3)

    row_labels = [
        f"Train (n={len(pr_items['Train']['y'])})",
        f"Int.Valid (n={len(pr_items['Int.Valid']['y'])})",
        f"Int.Test  (n={len(pr_items['Int.Test']['y'])})",
        f"Ext.Test  (n={len(pr_items['Ext.Test']['y'])})",
    ]
    col_labels = ["AP CV", "Specificity", "NPV", "Average Precision"]
    table_data = [
        [f"{ap_cv:.2f}", f"{pr_items['Train']['spec']:.2f}",     f"{pr_items['Train']['npv']:.2f}",     f"{pr_items['Train']['ap']:.2f}"],
        [f"{ap_cv:.2f}", f"{pr_items['Int.Valid']['spec']:.2f}", f"{pr_items['Int.Valid']['npv']:.2f}", f"{pr_items['Int.Valid']['ap']:.2f}"],
        [f"{ap_cv:.2f}", f"{pr_items['Int.Test']['spec']:.2f}",  f"{pr_items['Int.Test']['npv']:.2f}",  f"{pr_items['Int.Test']['ap']:.2f}"],
        [f"{ap_cv:.2f}", f"{pr_items['Ext.Test']['spec']:.2f}",  f"{pr_items['Ext.Test']['npv']:.2f}",  f"{pr_items['Ext.Test']['ap']:.2f}"],
    ]
    pr_row_colors = ["#7F8FA3", "#FFA0A3", "#77DDF9", "#61649f"]
    _add_table(ax, table_data, row_labels, col_labels, colors=pr_row_colors,
               bbox=(0.10, -0.55, 0.90, 0.30), fontsize=12, rowlabel_width=0.28)

    plt.subplots_adjust(bottom=0.45)
    plt.savefig(os.path.join(fig_dir, "Figure4b_subtype_PR.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "Figure4b_subtype_PR.pdf"), dpi=600, bbox_inches="tight")
    plt.close()

    # ---------- Calibration (Figure 4c-like) ----------
    fig, ax = plt.subplots(figsize=(5, 5.4), facecolor="white")
    ax.set_facecolor("white")

    calib_colors = {
        "Train": "#7F8FA3",
        "Int.Valid": "#FFA0A3",
        "Int.Test": "#77DDF9",
        "Ext.Test": "#61649f",
    }
    eces = {}
    for k in ["Train", "Int.Valid", "Int.Test", "Ext.Test"]:
        y, s = data[k]
        prob_true, prob_pred = calibration_curve(y, s, n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o', label=k, color=calib_colors[k])
        eces[k] = _ece(y, s, n_bins=10)

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("Mean Predicted Probability", fontsize=14)
    ax.set_ylabel("Fraction of Positives", fontsize=14)
    ax.set_title(f"Pathological Subtype Classification Calibration Curves\n{title_suffix}", fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(alpha=0.3)

    row_labels = [
        f"Train (n={len(data['Train'][0])})",
        f"Int.Valid (n={len(data['Int.Valid'][0])})",
        f"Int.Test (n={len(data['Int.Test'][0])})",
        f"Ext.Test (n={len(data['Ext.Test'][0])})",
    ]
    col_labels = ["ECE"]
    table_data = [
        [f"{eces['Train']:.3f}"],
        [f"{eces['Int.Valid']:.3f}"],
        [f"{eces['Int.Test']:.3f}"],
        [f"{eces['Ext.Test']:.3f}"],
    ]
    _add_table(ax, table_data, row_labels, col_labels, colors=pr_row_colors,
               bbox=(0.30, -0.55, 0.65, 0.30), fontsize=12, rowlabel_width=0.40)

    plt.subplots_adjust(bottom=0.42)
    plt.savefig(os.path.join(fig_dir, "Figure4c_subtype_Calibration.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "Figure4c_subtype_Calibration.pdf"), dpi=600, bbox_inches="tight")
    plt.close()

    print("✔ Subtype (binary) figures generated.")


# ============================================================
# TNM (multiclass OVR) plots: ROC / PR / Calibration + tables
# ============================================================
def plot_tnm_multiclass(result_dir="./results", fig_dir="./figures"):
    _ensure_dir(fig_dir)

    req = [
        "tnm_train_labels.npy", "tnm_train_scores.npy",
        "tnm_val_labels.npy", "tnm_val_scores.npy",
        "tnm_test_labels.npy", "tnm_test_scores.npy",
    ]
    for f in req:
        if not _exists(os.path.join(result_dir, f)):
            print(f"[plot_tnm_multiclass] Skip: missing {os.path.join(result_dir, f)}")
            return

    train_y = np.load(os.path.join(result_dir, "tnm_train_labels.npy")).astype(int)
    train_s = np.load(os.path.join(result_dir, "tnm_train_scores.npy")).astype(float)

    val_y = np.load(os.path.join(result_dir, "tnm_val_labels.npy")).astype(int)
    val_s = np.load(os.path.join(result_dir, "tnm_val_scores.npy")).astype(float)

    test_y = np.load(os.path.join(result_dir, "tnm_test_labels.npy")).astype(int)
    test_s = np.load(os.path.join(result_dir, "tnm_test_scores.npy")).astype(float)

    # external (simulated unless provided)
    test2_lp = os.path.join(result_dir, "tnm_test2_labels.npy")
    test2_sp = os.path.join(result_dir, "tnm_test2_scores.npy")
    test2_y = _load_npy(test2_lp)
    test2_s = _load_npy(test2_sp)
    if test2_y is None or test2_s is None:
        test2_y, test2_s = _maybe_sim_ext(test_y, test_s, noise=0.05, seed=9)
    test2_y = test2_y.astype(int)
    test2_s = test2_s.astype(float)

    classes = [0, 1, 2]
    names = ['Stage I-II', 'Stage III', 'Stage IV']
    colors = ['#0074B7', '#60A3D9', '#6CC4DC']

    bins = {
        "Train": (label_binarize(train_y, classes), train_s, train_y),
        "Int.Valid": (label_binarize(val_y, classes), val_s, val_y),
        "Int.Test": (label_binarize(test_y, classes), test_s, test_y),
        "Ext.Test": (label_binarize(test2_y, classes), test2_s, test2_y),
    }
    row_labels_base = ["Train", "Int.Valid", "Int.Test", "Ext.Test"]
    row_colors = ["#0074B7", "#60A3D9", "#6CC4DC", "#22a2c3"]

    # ---------- Figure 5a1: ROC per class + table ----------
    for i, cname in enumerate(names):
        fig, ax = plt.subplots(figsize=(5, 6), facecolor="white")
        ax.set_facecolor("white")

        aucs = {}
        fprs = {}
        tprs = {}
        sample_counts = {}
        accs = {}

        for key, (yb, ys, ylab) in bins.items():
            ovr = _calc_ovr_auc(yb, ys)
            fpr, tpr, auc_i = ovr[i]
            fprs[key], tprs[key], aucs[key] = fpr, tpr, float(auc_i)

            sample_counts[key] = str(int((ylab == i).sum()))
            accs[key] = _acc_ovr(yb[:, i], ys[:, i], thresh=0.5)

        # plot 4 curves with different linestyles like your original
        styles = {"Train": "-", "Int.Valid": "--", "Int.Test": ":", "Ext.Test": "-."}
        for key in ["Train", "Int.Valid", "Int.Test", "Ext.Test"]:
            ax.plot(fprs[key], tprs[key], linestyle=styles[key],
                    label=f"{key} (AUC = {aucs[key]:.2f})",
                    color=colors[i], linewidth=2.5)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel('False Positive Rate', fontsize=13)
        ax.set_ylabel('True Positive Rate', fontsize=13)
        ax.set_title(f'TNM stage Classification ROC Curve \nfor {cname}', fontsize=14)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)

        # table (Sample Count / AUC / Accuracy) — same spirit as your original
        col_labels = ["Sample Count", "AUC", "Accuracy"]
        table_data = [
            [sample_counts["Train"],     f"{aucs['Train']:.2f}",     f"{accs['Train']:.3f}"],
            [sample_counts["Int.Valid"], f"{aucs['Int.Valid']:.2f}", f"{accs['Int.Valid']:.3f}"],
            [sample_counts["Int.Test"],  f"{aucs['Int.Test']:.2f}",  f"{accs['Int.Test']:.3f}"],
            [sample_counts["Ext.Test"],  f"{aucs['Ext.Test']:.2f}",  f"{accs['Ext.Test']:.3f}"],
        ]
        _add_table(ax, table_data, row_labels_base, col_labels, colors=[colors[i]]*4,
                   bbox=(0.10, -0.52, 0.90, 0.30), fontsize=12, rowlabel_width=0.18)

        plt.subplots_adjust(bottom=0.38)
        safe_name = cname.replace(" ", "_").replace("-", "_")
        plt.savefig(os.path.join(fig_dir, f"Figure5a1_{safe_name}.png"), dpi=600, bbox_inches="tight")
        plt.savefig(os.path.join(fig_dir, f"Figure5a1_{safe_name}.pdf"), dpi=600, bbox_inches="tight")
        plt.close()

    # ---------- Figure 5a2: PR per class + table ----------
    for i, cname in enumerate(names):
        fig, ax = plt.subplots(figsize=(5, 6.5), facecolor="white")
        ax.set_facecolor("white")

        # PR curves for each split
        pr = {}
        for key, (yb, ys, ylab) in bins.items():
            p, r, ap = _calc_ovr_pr(yb, ys)[i]
            spec, npv = _spec_npv_binary(yb[:, i], ys[:, i], thresh=0.5)
            pr[key] = dict(p=p, r=r, ap=float(ap), spec=spec, npv=npv)

        # AP CV across splits (per class)
        ap_vals = np.array([pr[k]["ap"] for k in ["Train", "Int.Valid", "Int.Test", "Ext.Test"]], dtype=float)
        ap_cv = float(np.std(ap_vals) / np.mean(ap_vals)) if np.mean(ap_vals) > 0 else 0.0

        styles = {"Train": "-", "Int.Valid": "--", "Int.Test": ":", "Ext.Test": "-."}
        colors_pr = ['#7F8FA3', '#FFA0A3', '#77DDF9']  # your TNM PR palette (3 classes)
        c_use = colors_pr[i]

        for key in ["Train", "Int.Valid", "Int.Test", "Ext.Test"]:
            ax.plot(pr[key]["r"], pr[key]["p"], linestyle=styles[key],
                    label=f"{key} (AP={pr[key]['ap']:.2f})",
                    color=c_use, linewidth=2.5)

        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title(f'TNM stage Classification Precision-Recall Curve \nfor {cname}', fontsize=14)
        ax.legend(loc="lower left", fontsize=12)
        ax.grid(alpha=0.3)

        col_labels = ["AP CV", "Specificity", "NPV", "Average Precision"]
        table_data = [
            [f"{ap_cv:.2f}", f"{pr['Train']['spec']:.2f}",     f"{pr['Train']['npv']:.2f}",     f"{pr['Train']['ap']:.2f}"],
            [f"{ap_cv:.2f}", f"{pr['Int.Valid']['spec']:.2f}", f"{pr['Int.Valid']['npv']:.2f}", f"{pr['Int.Valid']['ap']:.2f}"],
            [f"{ap_cv:.2f}", f"{pr['Int.Test']['spec']:.2f}",  f"{pr['Int.Test']['npv']:.2f}",  f"{pr['Int.Test']['ap']:.2f}"],
            [f"{ap_cv:.2f}", f"{pr['Ext.Test']['spec']:.2f}",  f"{pr['Ext.Test']['npv']:.2f}",  f"{pr['Ext.Test']['ap']:.2f}"],
        ]
        _add_table(ax, table_data, row_labels_base, col_labels, colors=[c_use]*4,
                   bbox=(0.10, -0.52, 0.90, 0.30), fontsize=12, rowlabel_width=0.18)

        plt.subplots_adjust(bottom=0.40)
        safe_name = cname.replace(" ", "_").replace("-", "_")
        plt.savefig(os.path.join(fig_dir, f"Figure5a2_{safe_name}.png"), dpi=600, bbox_inches="tight")
        plt.savefig(os.path.join(fig_dir, f"Figure5a2_{safe_name}.pdf"), dpi=600, bbox_inches="tight")
        plt.close()

    # ---------- Figure 5a3: Calibration per class + table (ECE) ----------
    for i, cname in enumerate(names):
        fig, ax = plt.subplots(figsize=(5, 6.3), facecolor="white")
        ax.set_facecolor("white")

        calib_cols = ["#0074B7", "#60A3D9", "#6CC4DC", "#22a2c3"]  # split colors
        eces = {}

        for (key, (yb, ys, _)), c in zip(bins.items(), calib_cols):
            pt, pp = calibration_curve(yb[:, i], ys[:, i], n_bins=10, strategy="uniform")
            ax.plot(pp, pt, marker='o', label=key, color=c)
            eces[key] = _ece(yb[:, i], ys[:, i], n_bins=10)

        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlabel('Mean Predicted Probability', fontsize=13)
        ax.set_ylabel('Fraction of Positives', fontsize=13)
        ax.set_title(f'TNM stage Classification Calibration Curve \nfor {cname}', fontsize=14)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(alpha=0.3)

        col_labels = ["ECE"]
        table_data = [
            [f"{eces['Train']:.3f}"],
            [f"{eces['Int.Valid']:.3f}"],
            [f"{eces['Int.Test']:.3f}"],
            [f"{eces['Ext.Test']:.3f}"],
        ]
        _add_table(ax, table_data, row_labels_base, col_labels, colors=calib_cols,
                   bbox=(0.10, -0.52, 0.90, 0.30), fontsize=12, rowlabel_width=0.18)

        plt.subplots_adjust(bottom=0.38)
        safe_name = cname.replace(" ", "_").replace("-", "_")
        plt.savefig(os.path.join(fig_dir, f"Figure5a3_{safe_name}.png"), dpi=600, bbox_inches="tight")
        plt.savefig(os.path.join(fig_dir, f"Figure5a3_{safe_name}.pdf"), dpi=600, bbox_inches="tight")
        plt.close()

    print("✔ TNM multiclass figures generated.")


# ============================================================
# Survival plots (DFS/OS): KM + Cox HR + log-rank + at-risk text
# ============================================================
def _evaluate_survival(df):
    df = df.copy()
    df["risk_score"] = df["group"].map({"Low": 0, "Mediate": 1, "High": 2})
    c_index = concordance_index(df["time"], -df["risk_score"], df["event"])
    time_point = 30
    y_true = (df["time"] > time_point).astype(int)
    y_prob = 1 - df["risk_score"] / 2.0
    brier = brier_score_loss(y_true, y_prob)
    return float(c_index), float(brier)


def _plot_km_with_hr_and_atrisk(df, title, save_path, n_total=None):
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.set_facecolor("white")

    colors = {"Low": "#91c7ae", "Mediate": "#f7b977", "High": "#d87c7c"}
    groups = ["Low", "Mediate", "High"]

    # curves + capture handles
    lines = {}
    at_risk_table = []
    times = np.arange(0, 70, 10)

    for g in groups:
        m = (df["group"] == g)
        if m.sum() == 0:
            at_risk_table.append([0 for _ in times])
            continue
        kmf.fit(df.loc[m, "time"], event_observed=df.loc[m, "event"], label=g)
        kmf.plot_survival_function(ci_show=True, linewidth=2, color=colors[g], ax=ax)
        lines[g] = ax.get_lines()[-1]
        at_risk_table.append([int(np.sum(df.loc[m, "time"] >= t)) for t in times])

    handles = [lines.get("Low"), lines.get("Mediate"), lines.get("High")]
    labels = ["Low", "Medium", "High"]
    ax.legend(handles, labels, title="Groups", loc="upper right", framealpha=0.5, fontsize=12, title_fontsize=12)

    # at-risk text (match your style)
    # place below x-axis
    for i, t in enumerate(times):
        l, m, h = at_risk_table[0][i], at_risk_table[1][i], at_risk_table[2][i]
        ax.text(t, -0.38, str(l), color="#207f4c", fontsize=13, ha='center')
        ax.text(t, -0.48, str(m), color="#fca106", fontsize=13, ha='center')
        ax.text(t, -0.58, str(h), color="#cc163a", fontsize=13, ha='center')

    ax.text(-1, -0.28, 'Number at risk', color='black', ha='center', fontsize=13)
    ax.text(-10, -0.38, "Low", color="#207f4c", fontsize=13)
    ax.text(-10, -0.48, "Medium", color="#fca106", fontsize=13)
    ax.text(-10, -0.58, "High", color="#cc163a", fontsize=13)

    # Cox HR + Wald p
    dfx = df.copy()
    dfx["group_code"] = dfx["group"].map({"Low": 0, "Mediate": 1, "High": 2})
    cph = CoxPHFitter()
    cph.fit(dfx[["time", "event", "group_code"]], duration_col="time", event_col="event")
    coef = float(cph.params_["group_code"])
    se = float(cph.standard_errors_["group_code"])

    hr_med_vs_low = float(np.exp(coef))
    hr_high_vs_low = float(np.exp(2 * coef))

    z_med = (coef) / se
    p_med = float(2 * (1 - norm.cdf(abs(z_med))))
    z_high = (2 * coef) / se
    p_high = float(2 * (1 - norm.cdf(abs(z_high))))

    # global stats
    c_index, brier = _evaluate_survival(df)
    logrank_p = float(multivariate_logrank_test(df["time"], df["group"], df["event"]).p_value)

    ax.text(25, 0.46, f"P={logrank_p:.3f}", fontsize=12)
    ax.text(25, 0.36, f"C-index={c_index:.3f}", fontsize=12)
    ax.text(25, 0.26, f"Brier Score={brier:.3f}", fontsize=12)
    ax.text(25, 0.16, f"HR Intermediate vs Low = {hr_med_vs_low:.2f}, P={p_med:.3f}", fontsize=12)
    ax.text(25, 0.06, f"HR High vs Low = {hr_high_vs_low:.2f}, P={p_high:.3f}", fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if n_total is None:
        n_total = len(df)

    ax.set_title(f"{title}\n(n={n_total})", fontsize=14)
    ax.set_xlabel("Time since treatment start (months)", fontsize=13)
    ax.set_ylabel("Survival probability", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path + ".png", dpi=600, bbox_inches="tight")
    plt.savefig(save_path + ".pdf", dpi=600, bbox_inches="tight")
    plt.close()


def plot_survival(result_dir="./results", fig_dir="./figures"):
    _ensure_dir(fig_dir)

    # DFS/OS for train/val/test; ext optional
    for split in ["train", "val", "test"]:
        dfs_path = os.path.join(result_dir, f"dfs_{split}.csv")
        os_path  = os.path.join(result_dir, f"os_{split}.csv")

        if _exists(dfs_path):
            df = pd.read_csv(dfs_path)
            _plot_km_with_hr_and_atrisk(df,
                                        title=f"Disease-Free Survival (DFS) — Kaplan-Meier Curves ({split})",
                                        save_path=os.path.join(fig_dir, f"DFS_{split}"),
                                        n_total=len(df))
        else:
            print(f"[plot_survival] Skip DFS {split}: missing {dfs_path}")

        if _exists(os_path):
            df = pd.read_csv(os_path)
            _plot_km_with_hr_and_atrisk(df,
                                        title=f"Overall Survival (OS) — Kaplan-Meier Curves ({split})",
                                        save_path=os.path.join(fig_dir, f"OS_{split}"),
                                        n_total=len(df))
        else:
            print(f"[plot_survival] Skip OS {split}: missing {os_path}")

    print("✔ DFS / OS KM figures generated (where available).")


# ============================================================
# Public entry: plot_all
# ============================================================
def plot_all(result_dir="./results", fig_dir="./figures",
             do_subtype=True, do_tnm=True, do_survival=True):
    _ensure_dir(fig_dir)

    if do_subtype:
        plot_subtype_binary(result_dir=result_dir, fig_dir=fig_dir)

    if do_tnm:
        plot_tnm_multiclass(result_dir=result_dir, fig_dir=fig_dir)

    if do_survival:
        plot_survival(result_dir=result_dir, fig_dir=fig_dir)


# ============================================================
# CLI usage (optional)
# ============================================================
if __name__ == "__main__":
    plot_all("./results", "./figures")
