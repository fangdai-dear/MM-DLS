from torch.utils.data import DataLoader
from PatientDataset import PatientMultiModalDataset

def make_loader(
    split_dir: str,
    batch_size: int = 4,
    n_slices: int = 10,
    img_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
):
    ds = PatientMultiModalDataset(
        split_dir=split_dir,
        n_slices=n_slices,
        img_size=(img_size, img_size),
        clinical_dim=6,
        radiomics_dim=128,
        pet_dim=5,
        seed=0,
        require_space=True,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
