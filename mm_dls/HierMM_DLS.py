import torch
import torch.nn as nn
import torch.nn.functional as F

from mm_dls.ModelLesionEncoder import LesionEncoder
from mm_dls.ModelSpaceEncoder import SpaceEncoder
from mm_dls.LesionAttentionFusion import LesionAttentionFusion


class HierMM_DLS(nn.Module):
    """
    Hierarchical multi-task model:
      Stage-1: subtype classification + TNM classification
      Stage-2: survival Cox risks (DFS/OS) conditioned on subtype/TNM soft embeddings
      Stage-3: fixed-horizon binary classification (DFS/OS at 1y/3y/5y) logits

    Inputs:
      lesion_vol: [B,S,1,H,W]
      space_vol : [B,S,1,H,W]
      radiomics : [B,128]
      pet       : [B,5]
      clinical  : [B,C]

    Outputs:
      subtype_logits: [B, K_sub]
      tnm_logits    : [B, K_tnm]
      dfs_risk      : [B]
      os_risk       : [B]
      dfs_logits    : [B,3]  (1y,3y,5y)
      os_logits     : [B,3]
    """

    def __init__(
        self,
        num_subtypes: int,
        num_tnm: int,
        img_feat_dim: int = 128,
        radiomics_dim: int = 128,
        pet_dim: int = 5,
        clinical_dim: int = 6,
        task_emb_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lesion_encoder = LesionEncoder(input_channels=1, feature_dim=img_feat_dim)
        self.space_encoder  = SpaceEncoder(input_channels=1, feature_dim=img_feat_dim)

        self.lesion_fuser = LesionAttentionFusion(img_feat_dim, img_feat_dim)
        self.space_fuser  = LesionAttentionFusion(img_feat_dim, img_feat_dim)

        fused_base_dim = img_feat_dim * 2 + radiomics_dim + pet_dim + clinical_dim

        self.shared_up = nn.Sequential(
            nn.Linear(fused_base_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.subtype_head = nn.Linear(128, num_subtypes)
        self.tnm_head     = nn.Linear(128, num_tnm)

        self.subtype_emb = nn.Embedding(num_subtypes, task_emb_dim)
        self.tnm_emb     = nn.Embedding(num_tnm, task_emb_dim)

        surv_in = 128 + task_emb_dim * 2
        self.surv_mlp = nn.Sequential(
            nn.Linear(surv_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Cox risks
        self.dfs_head = nn.Linear(128, 1)
        self.os_head  = nn.Linear(128, 1)

        # Fixed-horizon classification logits (1y/3y/5y)
        self.dfs_cls = nn.Linear(128, 3)
        self.os_cls  = nn.Linear(128, 3)

    def _encode_volume(self, encoder, vol):
        # vol: [B,S,1,H,W]
        B, S, C, H, W = vol.shape
        x = vol.view(B * S, C, H, W)
        feat = encoder(x)          # [B*S, D]
        feat = feat.view(B, S, -1) # [B,S,D]
        return feat

    def forward(self, lesion_vol, space_vol, radiomics, pet, clinical):
        lesion_seq = self._encode_volume(self.lesion_encoder, lesion_vol)
        space_seq  = self._encode_volume(self.space_encoder,  space_vol)

        lesion_f = self.lesion_fuser(lesion_seq)  # [B,D]
        space_f  = self.space_fuser(space_seq)    # [B,D]

        base = torch.cat([lesion_f, space_f, radiomics, pet, clinical], dim=1)
        up = self.shared_up(base)                 # [B,128]

        subtype_logits = self.subtype_head(up)    # [B,Ks]
        tnm_logits     = self.tnm_head(up)        # [B,Kt]

        subtype_prob = F.softmax(subtype_logits, dim=1)
        tnm_prob     = F.softmax(tnm_logits, dim=1)

        subtype_e = subtype_prob @ self.subtype_emb.weight  # [B,E]
        tnm_e     = tnm_prob     @ self.tnm_emb.weight      # [B,E]

        surv_x = torch.cat([up, subtype_e, tnm_e], dim=1)
        surv_h = self.surv_mlp(surv_x)            # [B,128]

        dfs_risk = self.dfs_head(surv_h).squeeze(1)
        os_risk  = self.os_head(surv_h).squeeze(1)

        dfs_logits = self.dfs_cls(surv_h)         # [B,3]
        os_logits  = self.os_cls(surv_h)          # [B,3]

        return subtype_logits, tnm_logits, dfs_risk, os_risk, dfs_logits, os_logits
