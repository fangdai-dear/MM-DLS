import torch
import torch.nn as nn
import torch.nn.functional as F

class LesionAttentionFusion(nn.Module):
    def __init__(self, input_dim, output_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (input_dim // heads) ** 0.5
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, lesion_feat, lung_feat=None):
        """
        lesion_feat: [B, N, D] 或 [N, D] 单个病人时
        lung_feat:   [B, N, D] 或 [N, D]
        """
        if lung_feat is None:
            lung_feat = lesion_feat

        # 允许单个病人输入：自动添加 batch 维度
        added_batch = False
        if lesion_feat.dim() == 2:
            lesion_feat = lesion_feat.unsqueeze(0)  # -> [1, N, D]
            lung_feat = lung_feat.unsqueeze(0)
            added_batch = True

        B, N, D = lesion_feat.shape
        H = self.heads

        Q = self.q_proj(lesion_feat).view(B, N, H, -1).transpose(1, 2)  # [B, H, N, d]
        K = self.k_proj(lung_feat).view(B, N, H, -1).transpose(1, 2)   # [B, H, N, d]
        V = self.v_proj(lung_feat).view(B, N, H, -1).transpose(1, 2)   # [B, H, N, d]

        attn_weights = (Q @ K.transpose(-2, -1)) / self.scale
        attn_weights = self.dropout(F.softmax(attn_weights, dim=-1))  # [B, H, N, N]

        attn_output = attn_weights @ V  # [B, H, N, d]
        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)
        output = self.out_proj(attn_output) + lesion_feat  # residual connection

        # 做平均池化（每个病人输出一个 [D] 向量）
        output = output.mean(dim=1)  # [B, D]

        if added_batch:
            return output[0]  # 去掉 batch
        return output
