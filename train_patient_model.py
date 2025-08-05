import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import random

from ClinicalFusionModel import PatientLevelFusionModel
from CoxphLoss import CoxPHLoss
from LesionAttentionFusion import LesionAttentionFusion
from ModelLesionEncoder import LesionEncoder
from ModelSpaceEncoder import SpaceEncoder

class VariableSliceMockDataset(Dataset):
    def __init__(self, num_patients=900, max_slices=15, img_shape=(1, 64, 64)):
        self.max_slices = max_slices
        self.img_shape = img_shape
        self.data = []
        for _ in range(num_patients):
            n = random.randint(5, max_slices)
            lesion = torch.randn(n, *img_shape)
            space = torch.randn(n, *img_shape)
            lesion_pad = torch.cat([lesion, torch.zeros(max_slices - n, *img_shape)], dim=0)
            space_pad  = torch.cat([space,  torch.zeros(max_slices - n, *img_shape)], dim=0)
            self.data.append((lesion_pad, space_pad, n))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def generate_tabular_labels(num):
    return (
        torch.randn(num, 128),  # radiomics
        torch.randn(num, 5),    # PET
        torch.randn(num, 6),    # Clinical
        torch.rand(num) * 100,  # DFS time
        torch.randint(0, 2, (num,)).float(),  # DFS event
        torch.rand(num) * 150,  # OS time
        torch.randint(0, 2, (num,)).float(),  # OS event
        torch.cat([torch.zeros(num // 2), torch.ones(num - num // 2)]).float()[torch.randperm(num)]  # classification
    )

# 初始化数据
train_set = VariableSliceMockDataset(900)
valid_set = VariableSliceMockDataset(100)
test_set  = VariableSliceMockDataset(100)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=4)
test_loader  = DataLoader(test_set, batch_size=4)

tabular = {
    "train": generate_tabular_labels(900),
    "valid": generate_tabular_labels(100),
    "test":  generate_tabular_labels(100)
}

# 模型
lesion_encoder = LesionEncoder()
space_encoder = SpaceEncoder()
lesion_fuser = LesionAttentionFusion(128, 128)
space_fuser = LesionAttentionFusion(128, 128)
patient_model = PatientLevelFusionModel()

criterion_cls = nn.BCEWithLogitsLoss()
criterion_cox = CoxPHLoss()
optimizer = optim.Adam(
    list(lesion_encoder.parameters()) +
    list(space_encoder.parameters()) +
    list(lesion_fuser.parameters()) +
    list(space_fuser.parameters()) +
    list(patient_model.parameters()),
    lr=1e-4
)

# 评估函数
def evaluate_model(loader, tab_data):
    all_logits, all_labels, all_dfs, all_os = [], [], [], []
    all_dur_dfs, all_evt_dfs, all_dur_os, all_evt_os = [], [], [], []

    with torch.no_grad():
        for i, (lesion_slices, space_slices, n_valid) in enumerate(loader):
            B = lesion_slices.shape[0]
            idx_start = i * B
            idx_end = idx_start + B
            radiomics, pet, clinical, dur_dfs, evt_dfs, dur_os, evt_os, labels = [
                t[idx_start:idx_end] for t in tab_data
            ]

            lesion_features = []
            for b in range(B):
                encoded = lesion_encoder(lesion_slices[b, :n_valid[b]])  # [n_b, 128]
                fused = lesion_fuser(encoded.unsqueeze(0))               # [1, 128]
                lesion_features.append(fused)
            lesion_fused = torch.cat(lesion_features, dim=0)             # [B, 128]
            space_features = []
            for b in range(B):
                encoded = space_encoder(space_slices[b, :n_valid[b]])    # [n_b, 128]
                fused = space_fuser(encoded.unsqueeze(0))                # [1, 128]
                space_features.append(fused)
            space_fused = torch.cat(space_features, dim=0)               # [B, 128]
            # lesion_fused = lesion_fuser(lesion_encoded)
            # space_fused = space_fuser(space_encoded)
            dfs_pred, os_pred, cls_logits = patient_model(
                lesion_fused, space_fused, radiomics, pet, clinical
            )

            all_logits.append(cls_logits)
            all_labels.append(labels)
            all_dfs.append(dfs_pred)
            all_dur_dfs.append(dur_dfs)
            all_evt_dfs.append(evt_dfs)
            all_os.append(os_pred)
            all_dur_os.append(dur_os)
            all_evt_os.append(evt_os)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits)

    auc = roc_auc_score(labels.numpy(), probs.numpy()) if len(torch.unique(labels)) > 1 else 0.0
    acc = accuracy_score(labels.numpy(), (probs > 0.5).int().numpy())
    f1  = f1_score(labels.numpy(), (probs > 0.5).int().numpy())

    def concordance(pred, dur, evt):
        sorted_idx = torch.argsort(dur, descending=True)
        pred, dur, evt = pred[sorted_idx], dur[sorted_idx], evt[sorted_idx]
        n, correct = 0, 0
        for i in range(len(dur)):
            for j in range(i + 1, len(dur)):
                if evt[i] == 1 and dur[i] > dur[j]:
                    n += 1
                    correct += int(pred[i] > pred[j])
        return correct / n if n > 0 else 0.0

    c_dfs = concordance(torch.cat(all_dfs), torch.cat(all_dur_dfs), torch.cat(all_evt_dfs))
    c_os  = concordance(torch.cat(all_os),  torch.cat(all_dur_os),  torch.cat(all_evt_os))
    return auc, acc, f1, c_dfs, c_os

# 训练
for epoch in range(10):
    patient_model.train()
    for i, (lesion_slices, space_slices, n_valid) in enumerate(train_loader):
        B = lesion_slices.shape[0]
        idx_start = i * B
        idx_end = idx_start + B
        radiomics, pet, clinical, dur_dfs, evt_dfs, dur_os, evt_os, labels = [
            t[idx_start:idx_end] for t in tabular["train"]
        ]

        lesion_features = []
        space_features = []
        for b in range(B):
            lesion_feat = lesion_encoder(lesion_slices[b, :n_valid[b]])   # [n_b, 128]
            lesion_fused = lesion_fuser(lesion_feat.unsqueeze(0))         # [1, 128]
            lesion_features.append(lesion_fused)

            space_feat = space_encoder(space_slices[b, :n_valid[b]])      # [n_b, 128]
            space_fused = space_fuser(space_feat.unsqueeze(0))            # [1, 128]
            space_features.append(space_fused)

        lesion_fused = torch.cat(lesion_features, dim=0)  # [B, 128]
        space_fused = torch.cat(space_features, dim=0)    # [B, 128]

        dfs_pred, os_pred, cls_logits = patient_model(
            lesion_fused, space_fused, radiomics, pet, clinical
        )

        loss = (
            criterion_cox(dfs_pred, dur_dfs, evt_dfs) +
            criterion_cox(os_pred, dur_os, evt_os) +
            criterion_cls(cls_logits.squeeze(1), labels)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_metrics = evaluate_model(valid_loader, tabular["valid"])
    test_metrics = evaluate_model(test_loader, tabular["test"])
    print(f"Epoch {epoch+1} | Val: AUC={val_metrics[0]:.3f}, ACC={val_metrics[1]:.3f}, F1={val_metrics[2]:.3f}, "
          f"C-DFS={val_metrics[3]:.3f}, C-OS={val_metrics[4]:.3f} | "
          f"Test: AUC={test_metrics[0]:.3f}, ACC={test_metrics[1]:.3f}, F1={test_metrics[2]:.3f}, "
          f"C-DFS={test_metrics[3]:.3f}, C-OS={test_metrics[4]:.3f}")
