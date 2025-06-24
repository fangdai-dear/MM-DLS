import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict
import random

class CTMultiSlicePatientDataset(Dataset):
    def __init__(self, root_dir, n_slices=5, img_size=(64, 64)):
        self.n_slices = n_slices
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
        self.patient_dict = defaultdict(list)

        # 组织成 {patient_id: [slice_path1, ..., slice_pathN]}
        for path in sorted(glob(os.path.join(root_dir, "images", "*.png"))):
            fname = os.path.basename(path)
            pid = fname.split("_")[0]
            self.patient_dict[pid].append(path)

        # 只保留切片数 ≥ n_slices 的患者
        self.patient_ids = [pid for pid, slices in self.patient_dict.items() if len(slices) >= n_slices]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        slices = sorted(self.patient_dict[pid])
        start = random.randint(0, len(slices) - self.n_slices)
        selected = slices[start: start + self.n_slices]

        imgs = []
        for path in selected:
            img = Image.open(path).convert('L')
            imgs.append(self.transform(img))  # [1, H, W]

        volume = torch.stack(imgs, dim=0)  # [n_slices, 1, H, W]
        return volume, volume  # lesion_slices, space_slices
    
    
def load_all_datasets(base_dir, n_slices=5, img_size=(64, 64), batch_size=4):
    def load_set(name):
        return CTMultiSlicePatientDataset(os.path.join(base_dir, name), n_slices=n_slices, img_size=img_size)

    train_set = load_set("train")
    valid_set = load_set("valid")
    test_set  = load_set("test")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def simulate_tabular_labels(patient_ids, seed=42):
    torch.manual_seed(seed)
    num = len(patient_ids)
    simulated = {
        'radiomics': torch.randn(num, 128),
        'pet': torch.randn(num, 5),
        'clinical': torch.randn(num, 6),
        'dfs_time': torch.rand(num) * 100,
        'dfs_event': torch.randint(0, 2, (num,)).float(),
        'os_time': torch.rand(num) * 150,
        'os_event': torch.randint(0, 2, (num,)).float(),
        'label': torch.randint(0, 2, (num,)).float()
    }
    return simulated
