"""Box utilities written by the team (Chapter 4): IoU and NMS.

Boxes are numpy arrays in (x1, y1, x2, y2) pixel format, x2 > x1 and y2 > y1.
"""
import numpy as np


def iou_matrix(a, b):
    """IoU between every box of a (N,4) and every box of b (M,4) -> (N,M)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def iou(box_a, box_b):
    """IoU of two boxes."""
    return float(iou_matrix(box_a, box_b)[0, 0])


def nms(boxes, scores, iou_thr=0.5):
    """Greedy non-maximum suppression. Returns the kept indices, best score first."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        overlaps = iou_matrix(boxes[i], boxes[rest])[0]
        order = rest[overlaps <= iou_thr]
    return keep


def class_aware_nms(boxes, scores, labels, iou_thr=0.5):
    """NMS applied separately for each class."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    labels = np.asarray(labels)
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        k = nms(boxes[idx], np.asarray(scores)[idx], iou_thr)
        keep.extend(int(idx[j]) for j in k)
    return sorted(keep, key=lambda i: -scores[i])


def xyxy_to_yolo(box, img_w, img_h):
    """(x1,y1,x2,y2) pixels -> (cx,cy,w,h) normalised to [0,1] (YOLO label format)."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h)


def yolo_to_xyxy(box, img_w, img_h):
    cx, cy, w, h = box
    return ((cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h)
