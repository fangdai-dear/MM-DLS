# main.py
import torch
from torch.utils.data import DataLoader
from ClinicalFusionModel import PatientLevelFusionModel
from CoxphLoss import CoxPHLoss
from LesionAttentionFusion import LesionAttentionFusion
from ModelLesionEncoder import LesionEncoder
from ModelSpaceEncoder import SpaceEncoder


batch_size = 4
n_slices = 10

lesion_slices = torch.randn(batch_size, n_slices, 1, 32, 32, 32)  
space_slices = torch.randn(batch_size, n_slices, 1, 32, 32, 32)   
radiomics_feat = torch.randn(batch_size, 128)  
pet_feat = torch.randn(batch_size, 5)       
clinical_feat = torch.randn(batch_size, 6)    


lesion_encoder = LesionEncoder()
space_encoder = SpaceEncoder()
lesion_fuser = LesionAttentionFusion(input_dim=128, output_dim=128)
space_fuser = LesionAttentionFusion(input_dim=128, output_dim=128)
patient_model = PatientLevelFusionModel()


lesion_encoded = torch.stack([lesion_encoder(lesion_slices[:, i]) for i in range(n_slices)], dim=1)
space_encoded = torch.stack([space_encoder(space_slices[:, i]) for i in range(n_slices)], dim=1)

lesion_fused = lesion_fuser(lesion_encoded)
space_fused = space_fuser(space_encoded)


dfs_pred, os_pred = patient_model(lesion_fused, space_fused, radiomics_feat, pet_feat, clinical_feat)

print("DFS prediction:", dfs_pred.shape)
print("OS prediction:", os_pred.shape)
