#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-only CNN + Cox baseline (patient-level).
- Input: clinical.csv + CT slices per patient (optionally with mask)
- Output: risk scores + metrics (C-index; optional time-dependent AUC if available)
- Uses a lightweight ResNet18 encoder + pooling + linear risk head.
- Trains with Cox partial likelihood loss.

Expected CT folder structure (example):
DATA_ROOT/
  patient001/
    ct/
      0001.png
      0002.png
    mask/   (optional)
      0001.png
      0002.png
  patient002/
    ct/...

Example:
python baselines/ct_cnn_survival.py \
  --data_root data/CT_ROOT \
  --clinical_csv data/clinical.csv \
  --endpoint DFS \
  --id_col patient_id \
  --time_col dfs_time \
  --event_col dfs_event \
  --output_dir results/ct_cnn_survival
"""

import os
import json
import argparse
import random
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models

from lifelines.utils import concordance_index


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def list_images(folder: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    if not os.path.isdir(folder):
        return []
    fns = [os.path.join(folder, x) for x in os.listdir(folder) if x.lower().endswith(exts)]
    return sorted(fns)


class PatientCTDataset(Dataset):
    """
    Each item: one patient -> returns a tensor of shape (S, 3, H, W)
    and survival labels (time, event).
    """
    def __init__(
        self,
        data_root: str,
        clin_df: pd.DataFrame,
        id_col: str,
        time_col: str,
        event_col: str,
        ct_subdir: str = "ct",
        max_slices: int = 32,
        image_size: int = 224,
    ):
        self.data_root = data_root
        self.clin = clin_df.reset_index(drop=True)
        self.id_col = id_col
        self.time_col = time_col
        self.event_col = event_col
        self.ct_subdir = ct_subdir
        self.max_slices = max_slices

        self.tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            # keep default normalization mild; reviewers can reproduce
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ])

    def __len__(self):
        return len(self.clin)

    def __getitem__(self, idx: int):
        row = self.clin.iloc[idx]
        pid = str(row[self.id_col])
        t = float(row[self.time_col])
        e = int(row[self.event_col])

        ct_dir = os.path.join(self.data_root, pid, self.ct_subdir)
        imgs = list_images(ct_dir)
        if len(imgs) == 0:
            raise FileNotFoundError(f"No CT images found for patient {pid} at: {ct_dir}")

        # sample up to max_slices uniformly
        if len(imgs) > self.max_slices:
            inds = np.linspace(0, len(imgs) - 1, self.max_slices).round().astype(int)
            imgs = [imgs[i] for i in inds]

        slices = []
        for fp in imgs:
            im = Image.open(fp).convert("RGB")
            slices.append(self.tf(im))
        x = torch.stack(slices, dim=0)  # (S,3,H,W)

        return {
            "patient_id": pid,
            "x": x,
            "time": torch.tensor(t, dtype=torch.float32),
            "event": torch.tensor(e, dtype=torch.long),
        }


def collate_fn(batch: List[Dict]):
    # variable S: pad to max S in batch
    pids = [b["patient_id"] for b in batch]
    times = torch.stack([b["time"] for b in batch], dim=0)
    events = torch.stack([b["event"] for b in batch], dim=0)

    max_s = max(b["x"].shape[0] for b in batch)
    xs = []
    masks = []
    for b in batch:
        x = b["x"]
        s = x.shape[0]
        if s < max_s:
            pad = torch.zeros((max_s - s, ) + x.shape[1:], dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
        xs.append(x)
        m = torch.zeros((max_s,), dtype=torch.bool)
        m[:s] = True
        masks.append(m)
    xs = torch.stack(xs, dim=0)      # (B,S,3,H,W)
    masks = torch.stack(masks, dim=0)  # (B,S)
    return pids, xs, masks, times, events


class CTEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = False, out_dim: int = 256):
        super().__init__()
        if backbone == "resnet18":
            net = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
            self.backbone = net
        else:
            raise ValueError("Only resnet18 is supported in this baseline.")

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):  # x: (N,3,H,W)
        f = self.backbone(x)     # (N, feat_dim)
        z = self.proj(f)         # (N, out_dim)
        return z


class SurvivalHead(nn.Module):
    def __init__(self, in_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z):  # (B, in_dim)
        return self.fc(z).squeeze(-1)  # (B,)


def masked_mean_pool(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    seq: (B,S,D)
    mask: (B,S) True for valid
    """
    mask_f = mask.float().unsqueeze(-1)  # (B,S,1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)  # (B,1)
    return (seq * mask_f).sum(dim=1) / denom


def cox_partial_likelihood_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """
    Cox partial likelihood (negative) with Efron not used (simple baseline).
    risk: (B,) higher risk => shorter survival
    time: (B,)
    event: (B,) 1=event, 0=censored
    """
    # sort by time descending
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    time = time[order]
    event = event[order].float()

    # log-sum-exp over risk set
    log_cum_sum_exp = torch.logcumsumexp(risk, dim=0)
    loss = -(risk - log_cum_sum_exp) * event
    return loss.sum() / (event.sum().clamp_min(1.0))


def try_time_dependent_auc(times: np.ndarray, events: np.ndarray, risks: np.ndarray, eval_times: list):
    try:
        from sksurv.metrics import cumulative_dynamic_auc
        from sksurv.util import Surv
    except Exception:
        return None

    y = Surv.from_arrays(events.astype(bool), times.astype(float))
    aucs, mean_auc = cumulative_dynamic_auc(y, y, risks, np.array(eval_times, dtype=float))
    out = {str(t): float(a) for t, a in zip(eval_times, aucs)}
    out["mean_auc"] = float(mean_auc)
    return out


@torch.no_grad()
def evaluate(model_enc, model_head, loader, device) -> Dict:
    model_enc.eval()
    model_head.eval()
    all_time, all_event, all_risk, all_pid = [], [], [], []

    for pids, xs, masks, times, events in loader:
        xs = xs.to(device)         # (B,S,3,H,W)
        masks = masks.to(device)   # (B,S)
        times = times.cpu().numpy()
        events_np = events.cpu().numpy()

        B, S = xs.shape[0], xs.shape[1]
        x_flat = xs.view(B * S, *xs.shape[2:])  # (B*S,3,H,W)

        z_flat = model_enc(x_flat)              # (B*S,D)
        z = z_flat.view(B, S, -1)               # (B,S,D)
        z_pat = masked_mean_pool(z, masks)      # (B,D)

        risk = model_head(z_pat).detach().cpu().numpy()

        all_pid += list(pids)
        all_time.append(times)
        all_event.append(events_np)
        all_risk.append(risk)

    all_time = np.concatenate(all_time)
    all_event = np.concatenate(all_event).astype(int)
    all_risk = np.concatenate(all_risk).astype(float)

    cidx = concordance_index(all_time, -all_risk, all_event)  # use -risk for concordance_index convention
    return {
        "cindex": float(cidx),
        "patient_id": np.array(all_pid),
        "time": all_time,
        "event": all_event,
        "risk": all_risk,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Root folder containing patient CT folders.")
    parser.add_argument("--clinical_csv", required=True)
    parser.add_argument("--endpoint", default="DFS", choices=["DFS", "OS"])
    parser.add_argument("--id_col", default="patient_id")
    parser.add_argument("--time_col", default=None)
    parser.add_argument("--event_col", default=None)
    parser.add_argument("--split_col", default=None, help="Optional split column in clinical CSV (train/val/test).")
    parser.add_argument("--train_split_value", default="train")
    parser.add_argument("--test_split_value", default="test")
    parser.add_argument("--ct_subdir", default="ct")
    parser.add_argument("--max_slices", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_times", type=str, default="1,3,5", help="Comma-separated years for AUC (optional).")
    parser.add_argument("--time_unit_in_years", action="store_true",
                        help="If set, time is in years; otherwise assume days and convert years->days for eval_times.")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clin = pd.read_csv(args.clinical_csv)
    if args.time_col is None:
        args.time_col = "dfs_time" if args.endpoint == "DFS" else "os_time"
    if args.event_col is None:
        args.event_col = "dfs_event" if args.endpoint == "DFS" else "os_event"

    for c in [args.id_col, args.time_col, args.event_col]:
        if c not in clin.columns:
            raise ValueError(f"Missing column in clinical_csv: {c}")

    if args.split_col and args.split_col in clin.columns:
        train_df = clin[clin[args.split_col].astype(str).str.lower() == str(args.train_split_value).lower()].copy()
        test_df = clin[clin[args.split_col].astype(str).str.lower() == str(args.test_split_value).lower()].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("Empty train/test split. Check split_col values.")
    else:
        # fallback: use all as both train and test (still runnable, but not a strict benchmark)
        train_df = clin.copy()
        test_df = clin.copy()

    train_ds = PatientCTDataset(
        data_root=args.data_root,
        clin_df=train_df,
        id_col=args.id_col,
        time_col=args.time_col,
        event_col=args.event_col,
        ct_subdir=args.ct_subdir,
        max_slices=args.max_slices,
        image_size=args.image_size,
    )
    test_ds = PatientCTDataset(
        data_root=args.data_root,
        clin_df=test_df,
        id_col=args.id_col,
        time_col=args.time_col,
        event_col=args.event_col,
        ct_subdir=args.ct_subdir,
        max_slices=args.max_slices,
        image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )

    enc = CTEncoder(backbone="resnet18", pretrained=False, out_dim=256).to(device)
    head = SurvivalHead(in_dim=256).to(device)

    optim = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    best_cindex = -1.0
    best_path = os.path.join(args.output_dir, f"best_ctcnn_{args.endpoint.lower()}.pt")

    for epoch in range(1, args.epochs + 1):
        enc.train()
        head.train()
        epoch_loss = 0.0
        n_batches = 0

        for _, xs, masks, times, events in train_loader:
            xs = xs.to(device)           # (B,S,3,H,W)
            masks = masks.to(device)     # (B,S)
            times_t = times.to(device)   # (B,)
            events_t = events.to(device) # (B,)

            B, S = xs.shape[0], xs.shape[1]
            x_flat = xs.view(B * S, *xs.shape[2:])

            z_flat = enc(x_flat)                 # (B*S,D)
            z = z_flat.view(B, S, -1)            # (B,S,D)
            z_pat = masked_mean_pool(z, masks)   # (B,D)
            risk = head(z_pat)                   # (B,)

            loss = cox_partial_likelihood_loss(risk, times_t, events_t)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        eval_res = evaluate(enc, head, test_loader, device)
        cidx = eval_res["cindex"]
        avg_loss = epoch_loss / max(n_batches, 1)

        print(f"Epoch {epoch:03d} | loss={avg_loss:.6f} | test_cindex={cidx:.4f}")

        if cidx > best_cindex:
            best_cindex = cidx
            torch.save({
                "encoder": enc.state_dict(),
                "head": head.state_dict(),
                "args": vars(args),
                "best_cindex": best_cindex
            }, best_path)

    # Load best and export predictions
    ckpt = torch.load(best_path, map_location=device)
    enc.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])

    final_res = evaluate(enc, head, test_loader, device)

    # Optional time-dependent AUC
    eval_years = [float(x) for x in args.eval_times.split(",") if x.strip() != ""]
    eval_times = eval_years if args.time_unit_in_years else [y * 365.0 for y in eval_years]
    auc_info = try_time_dependent_auc(final_res["time"], final_res["event"], final_res["risk"], eval_times)

    pred_df = pd.DataFrame({
        args.id_col: final_res["patient_id"],
        "time": final_res["time"],
        "event": final_res["event"],
        "risk": final_res["risk"],
    })
    pred_csv = os.path.join(args.output_dir, f"{args.endpoint.lower()}_predictions.csv")
    pred_df.to_csv(pred_csv, index=False)

    metrics = {
        "endpoint": args.endpoint,
        "device": str(device),
        "n_train": int(len(train_ds)),
        "n_test": int(len(test_ds)),
        "best_cindex_test": float(best_cindex),
        "time_dependent_auc": auc_info,
        "note": "CT-only CNN baseline with Cox partial likelihood loss. Predictions are patient-level risks."
    }
    with open(os.path.join(args.output_dir, f"{args.endpoint.lower()}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[CT-CNN Survival] Done.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {best_path}")
    print(f"Saved: {pred_csv}")


if __name__ == "__main__":
    main()
