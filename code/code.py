import os
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

# --- Dataset ---
class CTMaskFolderDataset(Dataset):
    def __init__(self, image_dir, lung_dir, lesion_dir, transform=None):
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.png')))
        self.lung_paths = sorted(glob(os.path.join(lung_dir, '*.png')))
        self.lesion_paths = sorted(glob(os.path.join(lesion_dir, '*.png')))
        assert len(self.image_paths) > 0, f"No images found in {image_dir}"
        assert len(self.image_paths) == len(self.lung_paths) == len(self.lesion_paths), "Mismatch in dataset size."
        self.transform = transform or transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        lung = Image.open(self.lung_paths[idx]).convert("L")
        lesion = Image.open(self.lesion_paths[idx]).convert("L")
        return self.transform(image), self.transform(lung), self.transform(lesion)

# --- Cross Attention Block ---
class LungToLesionAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** 0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, lesion_feat, lung_feat):
        B, N, D = lesion_feat.shape
        H = self.heads
        Q = self.q_proj(lesion_feat).view(B, N, H, -1).transpose(1, 2)
        K = self.k_proj(lung_feat).view(B, N, H, -1).transpose(1, 2)
        V = self.v_proj(lung_feat).view(B, N, H, -1).transpose(1, 2)
        attn_weights = (Q @ K.transpose(-2, -1)) / self.scale
        attn_weights = self.dropout(F.softmax(attn_weights, dim=-1))
        attn_output = attn_weights @ V
        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(attn_output) + lesion_feat

# --- Encoder ---
class SharedResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2
        )

    def forward(self, x):
        return self.encoder(x)

# --- Decoder ---
class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1)
        )

    def forward(self, x):
        return self.decode(x)

# --- Model ---
class LungLesionSegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SharedResNetEncoder()
        self.cross_attn = LungToLesionAttention(dim=128)
        self.lung_decoder = SimpleDecoder(128, 1)
        self.lesion_decoder = SimpleDecoder(128, 1)

    def forward(self, x):
        feat = self.encoder(x)
        B, C, H, W = feat.shape
        H_in, W_in = x.shape[2:]
        feat_flat = feat.view(B, C, -1).permute(0, 2, 1)
        lesion_feat_updated = self.cross_attn(feat_flat, feat_flat)
        feat_lung = feat_flat.permute(0, 2, 1).view(B, C, H, W)
        feat_lesion = lesion_feat_updated.permute(0, 2, 1).view(B, C, H, W)
        lung_mask = F.interpolate(torch.sigmoid(self.lung_decoder(feat_lung)), size=(H_in, W_in), mode='bilinear', align_corners=False)
        lesion_mask = F.interpolate(torch.sigmoid(self.lesion_decoder(feat_lesion)), size=(H_in, W_in), mode='bilinear', align_corners=False)
        return lung_mask, lesion_mask

# --- Dice Metrics ---
def dice_loss(pred, target, smooth=1):
    if pred.size() != target.size():
        pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    return 1 - dice.mean()

def dice_coef(pred, target, threshold=0.5):
    if pred.size() != target.size():
        pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(2, 3))
    return (2 * intersection) / (pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-8)

# --- Dataloader Setup ---
def get_dataloaders(root_dir, batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    train_path = os.path.join(root_dir, "train/images")
    if len(glob(os.path.join(train_path, '*.png'))) == 0:
        raise ValueError(f"No training data found in {train_path}")

    train_set = CTMaskFolderDataset(
        os.path.join(root_dir, "train/images"),
        os.path.join(root_dir, "train/lung_masks"),
        os.path.join(root_dir, "train/lesion_masks"),
        transform
    )
    val_set = CTMaskFolderDataset(
        os.path.join(root_dir, "valid/images"),
        os.path.join(root_dir, "valid/lung_masks"),
        os.path.join(root_dir, "valid/lesion_masks"),
        transform
    )
    test_set = CTMaskFolderDataset(
        os.path.join(root_dir, "test/images"),
        os.path.join(root_dir, "test/lung_masks"),
        os.path.join(root_dir, "test/lesion_masks"),
        transform
    )
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size)
    )

# --- Training Script ---
if __name__ == "__main__":
    root_path = "/export/home/daifang/lunghospital/mask/dataset"  # TODO: Replace with actual dataset root
    model = LungLesionSegmentor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_loader, val_loader, test_loader = get_dataloaders(root_path, batch_size=8)

    for epoch in range(10):
        model.train()
        for images, lung_gt, lesion_gt in train_loader:
            lung_pred, lesion_pred = model(images)
            loss = dice_loss(lung_pred, lung_gt) + dice_loss(lesion_pred, lesion_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            dice_l, dice_s = 0, 0
            for images, lung_gt, lesion_gt in val_loader:
                lung_pred, lesion_pred = model(images)
                dice_l += dice_coef(lung_pred, lung_gt).mean().item()
                dice_s += dice_coef(lesion_pred, lesion_gt).mean().item()
            print(f"Epoch {epoch} | Dice Lung: {dice_l/len(val_loader):.3f} | Dice Lesion: {dice_s/len(val_loader):.3f}")
