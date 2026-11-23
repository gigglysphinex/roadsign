import torch
import pytest
from roadsign.model import RoadSignNet, architecture_table, LAYER_SPEC, INPUT_SHAPE
from roadsign.shapes import total_params, check_against_torch


def test_hand_computed_matches_pytorch():
    model = RoadSignNet(num_classes=43)
    rows = architecture_table()
    real, hand, match = check_against_torch(model, rows)
    assert match, f"Mismatch: PyTorch={real}, hand={hand}"


def test_forward_pass_shape():
    model = RoadSignNet(num_classes=43)
    x = torch.randn(4, *INPUT_SHAPE)
    out = model(x)
    assert out.shape == (4, 43)


def test_layer_dimensions():
    rows = architecture_table()
    shapes = {n: s for n, s, _ in rows}
    assert shapes["conv1"] == (32, 32, 32)
    assert shapes["pool1"] == (32, 16, 16)
    assert shapes["pool2"] == (64, 8, 8)
    assert shapes["pool3"] == (128, 4, 4)
    assert shapes["conv4"] == (128, 4, 4)
    assert shapes["gap"] == (128,)
    assert shapes["fc"] == (43,)


def test_parameter_counts_per_layer():
    rows = architecture_table()
    params = {n: p for n, _, p in rows}
    assert params["conv1"] == 3 * 32 * 3 * 3 + 32          # 896
    assert params["conv2"] == 32 * 64 * 3 * 3 + 64         # 18496
    assert params["conv3"] == 64 * 128 * 3 * 3 + 128       # 73856
    assert params["conv4"] == 128 * 128 * 3 * 3 + 128      # 147584
    assert params["fc"] == 128 * 43 + 43                    # 5547
    assert params["bn1"] == 64                              # 2 * 32
    assert params["bn2"] == 128                             # 2 * 64
    assert params["bn3"] == 256                             # 2 * 128
    assert params["bn4"] == 256                             # 2 * 128
    assert total_params(rows) == 247083
