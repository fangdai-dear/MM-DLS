# main.py
import torch
from torch.utils.data import DataLoader
from ClinicalFusionModel import PatientLevelFusionModel
from CoxphLoss import CoxPHLoss
from LesionAttentionFusion import LesionAttentionFusion
from ModelLesionEncoder import LesionEncoder
from ModelSpaceEncoder import SpaceEncoder

# 模拟一批病人数据
batch_size = 4
n_slices = 10

lesion_slices = torch.randn(batch_size, n_slices, 1, 32, 32, 32)  # 每个病人10张病灶图像
space_slices = torch.randn(batch_size, n_slices, 1, 32, 32, 32)   # 每个病人10张空间图像
radiomics_feat = torch.randn(batch_size, 128)  # 从PyRadiomics提取的肺部特征
pet_feat = torch.randn(batch_size, 5)         # SUVmax, SUVmean 等5项
clinical_feat = torch.randn(batch_size, 6)    # 年龄、性别、吸烟史等

# 模型实例化
lesion_encoder = LesionEncoder()
space_encoder = SpaceEncoder()
lesion_fuser = LesionAttentionFusion(input_dim=128, output_dim=128)
space_fuser = LesionAttentionFusion(input_dim=128, output_dim=128)
patient_model = PatientLevelFusionModel()

# 编码 + attention 融合
lesion_encoded = torch.stack([lesion_encoder(lesion_slices[:, i]) for i in range(n_slices)], dim=1)
space_encoded = torch.stack([space_encoder(space_slices[:, i]) for i in range(n_slices)], dim=1)

lesion_fused = lesion_fuser(lesion_encoded)
space_fused = space_fuser(space_encoded)

# 融合预测输出
dfs_pred, os_pred = patient_model(lesion_fused, space_fused, radiomics_feat, pet_feat, clinical_feat)

print("DFS prediction:", dfs_pred.shape)
print("OS prediction:", os_pred.shape)