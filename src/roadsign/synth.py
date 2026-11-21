"""Copy-paste synthetic data: paste GTSRB sign crops onto road backgrounds (enhancement 2).

Use ONLY training-split backgrounds and training-split sign crops. Never the official
test images. Pasted boxes never overlap a real annotation or each other.
"""
import cv2
import numpy as np

from .boxes import iou_matrix


def _feather_mask(h, w, margin=0.12):
    m = np.zeros((h, w), np.float32)
    my, mx = max(1, int(h * margin)), max(1, int(w * margin))
    m[my:h - my, mx:w - mx] = 1.0
    k = max(3, (min(h, w) // 6) | 1)
    return cv2.GaussianBlur(m, (k, k), 0)[..., None]


def paste_signs(bg, crops, labels, existing_boxes=(), rng=None, n_signs=3,
                size_range=(16, 80), max_tries=30):
    """Return (image, boxes, labels) with up to n_signs pasted signs.

    bg: HxWx3 uint8 BGR. crops: list of HxWx3 uint8 sign crops. labels: class id per crop.
    existing_boxes: real annotations of bg, to avoid covering them.
    """
    rng = rng or np.random.default_rng(0)
    out = bg.copy()
    H, W = out.shape[:2]
    taken = [tuple(b) for b in existing_boxes]
    boxes, out_labels = [], []
    for _ in range(n_signs):
        for _try in range(max_tries):
            i = int(rng.integers(len(crops)))
            s = int(rng.integers(size_range[0], size_range[1] + 1))
            if s >= min(H, W):
                continue
            x1 = int(rng.integers(0, W - s)); y1 = int(rng.integers(0, H - s))
            box = (x1, y1, x1 + s, y1 + s)
            if taken and (iou_matrix(box, np.array(taken)) > 0).any():
                continue
            patch = cv2.resize(crops[i], (s, s), interpolation=cv2.INTER_AREA).astype(np.float32)
            patch = patch * float(rng.uniform(0.6, 1.3)) + float(rng.uniform(-20, 20))  # lighting
            if rng.random() < 0.5:
                k = int(rng.choice([3, 5]))
                patch = cv2.GaussianBlur(patch, (k, k), 0)  # motion / defocus blur
            patch = np.clip(patch, 0, 255)
            m = _feather_mask(s, s)
            roi = out[y1:y1 + s, x1:x1 + s].astype(np.float32)
            out[y1:y1 + s, x1:x1 + s] = (m * patch + (1 - m) * roi).astype(np.uint8)
            taken.append(box); boxes.append(box); out_labels.append(labels[i])
            break
    return out, boxes, out_labels
