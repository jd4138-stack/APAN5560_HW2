# helper_lib/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FCNN", "SimpleCNN", "EnhancedCNN", "get_model"]

# -------------------------
# Fully Connected (baseline)
# -------------------------
class FCNN(nn.Module):
    """
    A simple MLP for 64x64 RGB images.
    Input: (B, 3, 64, 64) -> flatten to 12288 -> 512 -> 100 -> num_classes
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(3 * 64 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.mlp(x)

# -------------------------
# Spec-matching CNN
# -------------------------
class SimpleCNN(nn.Module):
    """
    Architecture (matches your requirement exactly):
    Input: 3x64x64
    Conv(3->16, k3, s1, p1) + ReLU
    MaxPool(2,2)
    Conv(16->32, k3, s1, p1) + ReLU
    MaxPool(2,2)
    Flatten
    FC(8192->100) + ReLU
    FC(100->num_classes)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # After two pools: 64 -> 32 -> 16, channels=32 -> 32*16*16 = 8192
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -------------------------
# A slightly stronger CNN
# -------------------------
class EnhancedCNN(nn.Module):
    """
    Adds BatchNorm/Dropout and a third conv block.
    Input: 3x64x64
    [Conv(3->16) + BN + ReLU] + MaxPool -> 32x32
    [Conv(16->32) + BN + ReLU] + MaxPool -> 16x16
    [Conv(32->64) + BN + ReLU] + MaxPool -> 8x8
    Flatten (64*8*8=4096) -> FC(4096->256) + ReLU + Dropout -> FC(256->num_classes)
    """
    def __init__(self, num_classes: int = 10, p_drop: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)   # -> 16 x 32 x 32
        x = self.block2(x)   # -> 32 x 16 x 16
        x = self.block3(x)   # -> 64 x 8 x 8
        x = torch.flatten(x, 1)
        return self.classifier(x)

# -------------------------
# Factory
# -------------------------
def get_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    Return a model instance by name.
    Valid options (case/space/underscore-insensitive):
      - "FCNN"
      - "CNN" or "SimpleCNN"
      - "EnhancedCNN"
    """
    key = "".join(str(model_name).split()).lower()  # normalize: remove spaces/underscores, lowercase
    aliases = {
        "fcnn": FCNN,
        "cnn": SimpleCNN,
        "simplecnn": SimpleCNN,
        "enhancedcnn": EnhancedCNN,
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Choose from: {', '.join(sorted(set(aliases.keys())))}"
        )
    return aliases[key](num_classes=num_classes)
