"""RoadSignNet: CNN for 32x32 traffic sign classification into 43 classes (Phase 2).

Architecture rationale:
- Input is only 32x32, so we cannot afford many pooling layers (each halves spatial dims).
- Three conv blocks (conv -> bn -> relu -> pool) reduce 32 -> 16 -> 8 -> 4.
- A fourth conv block without pooling keeps 4x4 spatial resolution.
- Global average pooling (GAP) collapses 4x4 -> 1x1, replacing a large FC layer.
- A single FC layer maps to 43 classes.
- Dropout before the classifier for regularisation.
- Total: ~125K parameters, suitable for embedded inference.
"""
import torch
import torch.nn as nn


class RoadSignNet(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        # Block 1: 3x32x32 -> 32x16x16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 32x16x16 -> 64x8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 64x8x8 -> 128x4x4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 128x4x4 -> 128x4x4 (no pooling, deepen features)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Head: GAP -> dropout -> FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# Hand-computed architecture table using shapes.py formulas.
LAYER_SPEC = [
    # Block 1
    {"name": "conv1",  "type": "conv",    "F": 32,  "K": 3, "S": 1, "P": 1},
    {"name": "bn1",    "type": "bn"},
    {"name": "relu1",  "type": "relu"},
    {"name": "pool1",  "type": "maxpool", "K": 2, "S": 2},
    # Block 2
    {"name": "conv2",  "type": "conv",    "F": 64,  "K": 3, "S": 1, "P": 1},
    {"name": "bn2",    "type": "bn"},
    {"name": "relu2",  "type": "relu"},
    {"name": "pool2",  "type": "maxpool", "K": 2, "S": 2},
    # Block 3
    {"name": "conv3",  "type": "conv",    "F": 128, "K": 3, "S": 1, "P": 1},
    {"name": "bn3",    "type": "bn"},
    {"name": "relu3",  "type": "relu"},
    {"name": "pool3",  "type": "maxpool", "K": 2, "S": 2},
    # Block 4
    {"name": "conv4",  "type": "conv",    "F": 128, "K": 3, "S": 1, "P": 1},
    {"name": "bn4",    "type": "bn"},
    {"name": "relu4",  "type": "relu"},
    # Head
    {"name": "gap",     "type": "gap"},
    {"name": "flatten", "type": "flatten"},
    {"name": "dropout", "type": "dropout"},
    {"name": "fc",      "type": "linear", "J": 43},
]

INPUT_SHAPE = (3, 32, 32)


def architecture_table():
    """Return the hand-computed layer table as a list of (name, output_shape, params)."""
    from .shapes import describe
    return describe(LAYER_SPEC, INPUT_SHAPE)


def print_table():
    """Pretty-print the architecture table."""
    rows = architecture_table()
    total = sum(r[2] for r in rows)
    print(f"{'Layer':<12} {'Output Shape':<18} {'Params':>8}")
    print("-" * 42)
    for name, shape, params in rows:
        shape_str = "x".join(str(s) for s in shape)
        print(f"{name:<12} {shape_str:<18} {params:>8,}")
    print("-" * 42)
    print(f"{'Total':<12} {'':<18} {total:>8,}")
    return total
