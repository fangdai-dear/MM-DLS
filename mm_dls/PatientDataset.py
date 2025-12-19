# mm_dls/PatientDataset.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class PatientDataset(Dataset):
    def __init__(
        self,
        data_root,
        clinical_csv,
        radiomics_npy,
        pet_npy,
        n_slices=30,
        img_size=224
    ):
        super().__init__()

        self.data_root = data_root
        self.df = pd.read_csv(clinical_csv)
        self.radiomics = np.load(radiomics_npy) 
        self.pet = np.load(pet_npy)              

        self.n_slices = n_slices

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def _load_slices(self, folder):
        files = sorted(os.listdir(folder))[: self.n_slices]
        imgs = []
        for f in files:
            img = Image.open(os.path.join(folder, f)).convert("L")
            imgs.append(self.transform(img))
        imgs = torch.stack(imgs, dim=0)  # [S,1,H,W]
        return imgs

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["pid"]

        # -------- images --------
        lesion_dir = os.path.join(self.data_root, "images", pid, "lesion")
        space_dir  = os.path.join(self.data_root, "images", pid, "space")

        lesion = self._load_slices(lesion_dir)
        space  = self._load_slices(space_dir)

        # -------- tabular --------
        radiomics = torch.tensor(self.radiomics[idx], dtype=torch.float32)
        pet = torch.tensor(self.pet[idx], dtype=torch.float32)
        clinical = torch.zeros(6)  

        # -------- labels --------
        y_sub = torch.tensor(row["subtype"], dtype=torch.long)
        y_tnm = torch.tensor(row["tnm_stage"], dtype=torch.long)

        dfs_time  = torch.tensor(row["dfs_time"], dtype=torch.float32)
        dfs_event = torch.tensor(row["dfs_event"], dtype=torch.float32)

        os_time  = torch.tensor(row["os_time"], dtype=torch.float32)
        os_event = torch.tensor(row["os_event"], dtype=torch.float32)

        # 1y / 3y / 5y
        dfs_1y = torch.tensor(row["dfs_time"] <= 12, dtype=torch.float32)
        dfs_3y = torch.tensor(row["dfs_time"] <= 36, dtype=torch.float32)
        dfs_5y = torch.tensor(row["dfs_time"] <= 60, dtype=torch.float32)

        os_1y = torch.tensor(row["os_time"] <= 12, dtype=torch.float32)
        os_3y = torch.tensor(row["os_time"] <= 36, dtype=torch.float32)
        os_5y = torch.tensor(row["os_time"] <= 60, dtype=torch.float32)

        treatment = torch.tensor(row["treatment"], dtype=torch.long)

        return (
            pid,
            lesion,
            space,
            radiomics,
            pet,
            clinical,
            y_sub,
            y_tnm,
            dfs_time,
            dfs_event,
            os_time,
            os_event,
            dfs_1y,
            dfs_3y,
            dfs_5y,
            os_1y,
            os_3y,
            os_5y,
            treatment,   
        )
