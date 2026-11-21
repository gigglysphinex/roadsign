"""Classical baseline with no deep learning (Chapter 1).

1. Classification of 32x32 crops: HOG + linear SVM.
2. Detection of sign candidates: HSV colour segmentation (red / blue) + contour and
   shape filtering. Measure its recall at IoU >= 0.5 on the VALIDATION split.
"""
import cv2
import numpy as np

from .boxes import iou_matrix


def hog_features(img_bgr, size=32):
    from skimage.feature import hog
    g = cv2.cvtColor(cv2.resize(img_bgr, (size, size)), cv2.COLOR_BGR2GRAY)
    return hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")


def train_hog_svm(imgs, labels, C=1.0):
    from sklearn.svm import LinearSVC
    X = np.stack([hog_features(i) for i in imgs])
    return LinearSVC(C=C).fit(X, labels)


def predict_hog_svm(model, imgs):
    return model.predict(np.stack([hog_features(i) for i in imgs]))


def color_shape_candidates(img_bgr, min_side=12, max_side=250, min_fill=0.35):
    """Return candidate boxes (x1,y1,x2,y2) from red/blue regions with plausible shape."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 90, 60), (10, 255, 255)) | cv2.inRange(hsv, (170, 90, 60), (180, 255, 255))
    blue = cv2.inRange(hsv, (100, 120, 50), (130, 255, 255))
    mask = cv2.morphologyEx(red | blue, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if not (min_side <= w <= max_side and min_side <= h <= max_side):
            continue
        if not 0.6 <= w / h <= 1.6:
            continue
        if cv2.contourArea(c) / (w * h) < min_fill:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def detection_recall(pred_boxes, gt_boxes, iou_thr=0.5):
    """Share of ground-truth boxes matched by at least one prediction, and precision."""
    if len(gt_boxes) == 0:
        return float("nan"), float("nan")
    if len(pred_boxes) == 0:
        return 0.0, float("nan")
    m = iou_matrix(np.array(gt_boxes), np.array(pred_boxes))
    recall = float((m.max(axis=1) >= iou_thr).mean())
    precision = float((m.max(axis=0) >= iou_thr).mean())
    return recall, precision
