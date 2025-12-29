import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageShotModel(nn.Module):
    """
    Simple CNN:
      Conv(3x3) -> ReLU -> Conv(3x3) -> ReLU -> GlobalAvgPool -> Linear -> Softmax
    Expects input: (N, C, H, W)
    """

    def __init__(self, in_channels: int = 3, output_dim: int = 2, hidden_channels: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 64x64 -> 32x32
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: 32x32 -> 16x16
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4: 16x16 -> 8x8
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Flatten: 8 * 8 * (hidden_channels * 8) = 64 * 256 = 16384
        self.flatten_dim = (hidden_channels * 8) * 8 * 8
        
        self.regressor = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        out = self.regressor(x)
        return out