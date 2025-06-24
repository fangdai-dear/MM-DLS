import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np

class PatientLevelFusionModel(nn.Module):
    def __init__(self, input_dim=128, pet_dim=5, clinical_dim=6):
        super().__init__()
        self.fc_merge = nn.Sequential(
            nn.Linear(input_dim * 2 + 128, 256),  # lesion_fused + space_fused + radiomics_feat
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        total_feat = 128 + pet_dim + clinical_dim
        self.fc_dfs = nn.Linear(total_feat, 1)
        self.fc_os = nn.Linear(total_feat, 1)
        self.fc_cls = nn.Linear(total_feat, 1)

    def forward(self, lesion_feat, space_feat, radiomics_feat, pet_feat, clinical_feat):
        x = torch.cat([lesion_feat, space_feat, radiomics_feat], dim=1)
        fused = self.fc_merge(x)  # shape [B, 128]
        full_feat = torch.cat([fused, pet_feat, clinical_feat], dim=1)
        dfs = self.fc_dfs(full_feat).squeeze(1)
        os = self.fc_os(full_feat).squeeze(1)
        cls = self.fc_cls(full_feat)  # keep [B, 1] for BCEWithLogits
        return dfs, os, cls

    @staticmethod
    def classification_metrics(logits, labels):
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        try:
            auc = roc_auc_score(labels, probs)
        except:
            auc = 0.0
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        return auc, acc, f1

    @staticmethod
    def c_index(preds, durations, events):
        preds = preds.detach().cpu().numpy()
        durations = durations.detach().cpu().numpy()
        events = events.detach().cpu().numpy()

        n = len(preds)
        num = 0
        den = 0
        for i in range(n):
            for j in range(i + 1, n):
                if durations[i] == durations[j]:
                    continue
                if events[i] == 1 and durations[i] < durations[j]:
                    den += 1
                    if preds[i] < preds[j]:
                        num += 1
                    elif preds[i] == preds[j]:
                        num += 0.5
                elif events[j] == 1 and durations[j] < durations[i]:
                    den += 1
                    if preds[j] < preds[i]:
                        num += 1
                    elif preds[j] == preds[i]:
                        num += 0.5
        return num / den if den > 0 else 0.0
