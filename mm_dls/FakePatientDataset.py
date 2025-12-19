import torch
from torch.utils.data import Dataset
import numpy as np
import random


class FakePatientDataset(Dataset):
    """
    Controllable synthetic multimodal + survival dataset

    You can explicitly control:
    - Final AUC (classification)
    - Final C-index (DFS / OS)
    via interpretable hyperparameters.

    Output: 19 items (aligned with run_epoch_verbose)
    """

    def __init__(
        self,
        n_patients=3000,
        n_slices=30,
        img_size=224,
        num_subtypes=2,
        num_tnm=3,
        seed=2131,

        # =========================
        # ---- AUC controllers ----
        # =========================
        tabular_signal_dims=16,        # ↑ dims → ↑ AUC
        tabular_signal_strength=0.40, # ↑ strength → ↑ AUC
        label_flip_rate=0.10,          # ↑ noise → ↓ AUC

        # =========================
        # ---- C-index controllers
        # =========================
        risk_noise=1.0,               # ↑ noise → ↓ C-index
        dfs_time_noise=6.0,
        os_time_noise=7.0,
        event_sharpness=1.3,          # ↑ → HR更明显
    ):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.n = n_patients
        self.n_slices = n_slices
        self.img_size = img_size
        self.num_subtypes = num_subtypes
        self.num_tnm = num_tnm

        self.tabular_signal_dims = tabular_signal_dims
        self.tabular_signal_strength = tabular_signal_strength
        self.label_flip_rate = label_flip_rate

        self.risk_noise = risk_noise
        self.dfs_time_noise = dfs_time_noise
        self.os_time_noise = os_time_noise
        self.event_sharpness = event_sharpness

        # =========================
        # Treatment cohort
        # =========================
        self.treatment = np.random.choice(
            [0, 1],
            size=self.n,
            p=[2374 / (2374 + 1790), 1790 / (2374 + 1790)]
        ).astype(np.int64)

        # =========================
        # Ground-truth labels
        # =========================
        self.subtype = np.random.randint(0, num_subtypes, size=self.n).astype(np.int64)
        self.tnm = np.random.randint(0, num_tnm, size=self.n).astype(np.int64)

        # =========================
        # Latent biological risk
        # =========================
        base_risk = (
            0.6 * self.subtype +
            0.5 * self.tnm +
            0.4 * self.treatment +
            np.random.normal(0, self.risk_noise, size=self.n)
        )

        # =========================
        # Survival times
        # =========================
        self.dfs_time = np.clip(
            60 - 7.0 * base_risk + np.random.normal(0, self.dfs_time_noise, size=self.n),
            3, 96
        )
        self.os_time = np.clip(
            75 - 8.5 * base_risk + np.random.normal(0, self.os_time_noise, size=self.n),
            6, 120
        )

        # =========================
        # Event indicators (soft)
        # =========================
        p_dfs = 1 / (1 + np.exp(-(base_risk - 0.2) * self.event_sharpness))
        p_os  = 1 / (1 + np.exp(-(base_risk - 0.4) * self.event_sharpness))

        self.dfs_event = (np.random.rand(self.n) < p_dfs).astype(np.float32)
        self.os_event  = (np.random.rand(self.n) < p_os).astype(np.float32)

        # =========================
        # Time-point labels
        # =========================
        self.dfs_1y = (self.dfs_time <= 12).astype(np.float32)
        self.dfs_3y = (self.dfs_time <= 36).astype(np.float32)
        self.dfs_5y = (self.dfs_time <= 60).astype(np.float32)

        self.os_1y  = (self.os_time <= 12).astype(np.float32)
        self.os_3y  = (self.os_time <= 36).astype(np.float32)
        self.os_5y  = (self.os_time <= 60).astype(np.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s = int(self.subtype[idx])
        t = int(self.tnm[idx])
        tr = int(self.treatment[idx])

        # =========================
        # Label noise (controls AUC ceiling)
        # =========================
        if np.random.rand() < self.label_flip_rate:
            s = 1 - s

        # =========================
        # IMAGE: very weak signal
        # =========================
        base_img = np.random.normal(0.5, 0.30, (self.img_size, self.img_size)).astype(np.float32)
        base_img += 0.03 * s + 0.02 * t + 0.02 * tr
        base_img = np.clip(base_img, 0, 1)

        lesion = torch.from_numpy(
            np.repeat(base_img[None, None, ...], self.n_slices, axis=0)
        )
        space = lesion.clone()

        # =========================
        # TABULAR: main discriminative signal
        # =========================
        radiomics = np.random.normal(0, 1.0, 128).astype(np.float32)
        radiomics[:self.tabular_signal_dims] += (
            self.tabular_signal_strength * s +
            0.7 * self.tabular_signal_strength * t +
            np.random.normal(0, 0.8, self.tabular_signal_dims)
        )

        pet = np.random.normal(0, 1.0, 5).astype(np.float32)
        pet[:2] += 0.5 * self.tabular_signal_strength * s + np.random.normal(0, 0.7, 2)

        clinical = np.random.normal(0, 1.0, 6).astype(np.float32)
        clinical[:3] += 0.5 * self.tabular_signal_strength * t + np.random.normal(0, 0.7, 3)

        return (
            f"P{idx:04d}",

            lesion.float(),
            space.float(),

            torch.from_numpy(radiomics),
            torch.from_numpy(pet),
            torch.from_numpy(clinical),

            torch.tensor(s, dtype=torch.long),
            torch.tensor(t, dtype=torch.long),

            torch.tensor(self.dfs_time[idx], dtype=torch.float32),
            torch.tensor(self.dfs_event[idx], dtype=torch.float32),

            torch.tensor(self.os_time[idx], dtype=torch.float32),
            torch.tensor(self.os_event[idx], dtype=torch.float32),

            torch.tensor(self.dfs_1y[idx], dtype=torch.float32),
            torch.tensor(self.dfs_3y[idx], dtype=torch.float32),
            torch.tensor(self.dfs_5y[idx], dtype=torch.float32),

            torch.tensor(self.os_1y[idx], dtype=torch.float32),
            torch.tensor(self.os_3y[idx], dtype=torch.float32),
            torch.tensor(self.os_5y[idx], dtype=torch.float32),

            torch.tensor(tr, dtype=torch.long),
        )
