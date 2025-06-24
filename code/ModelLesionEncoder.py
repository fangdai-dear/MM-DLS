import torch.nn as nn

class LesionEncoder(nn.Module):
    def __init__(self, input_channels=1, feature_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 输出 [B, 64, 1, 1]
            nn.Flatten(),                  # [B, 64]
            nn.Linear(64, feature_dim),    # → [B, 128]
            nn.ReLU(inplace=True)
        )

    def forward(self, x):  # x: [B, 1, H, W]
        return self.encoder(x)  # [B, 128]
