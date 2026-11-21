#!/usr/bin/env python3
"""Run the classical (no-DL) baseline on the GTSDB validation split (Phase 1, Ch.1).

Pipeline:
  1. Detection:  HSV colour + shape filtering  -> candidate boxes.
  2. Classification: HOG features + linear SVM  -> class label per crop.

Measures detection recall/precision at IoU >= 0.5, classification accuracy on
ground-truth crops, and speed (ms/image).  Saves metrics to JSON.

Usage:
    python scripts/run_baseline.py --config configs/phase1.yaml
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadsign.data import (
    parse_gtsdb_gt, load_split, assert_not_test, GTSDB_SIZE,
)
from roadsign.baseline import (
    hog_features, train_hog_svm, predict_hog_svm,
    color_shape_candidates, detection_recall,
)
from roadsign.boxes import iou_matrix


def load_image(gtsdb_root, name):
    assert_not_test(name)
    path = os.path.join(gtsdb_root, name)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    return img


def extract_gt_crops(img, boxes):
    """Cut ground-truth bounding boxes from an image, resize to 32x32."""
    crops = []
    for x1, y1, x2, y2 in boxes:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(cv2.resize(crop, (32, 32)))
    return crops


def main():
    parser = argparse.ArgumentParser(description="Classical baseline on GTSDB val")
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["baseline_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Load split and annotations.
    split_path = os.path.join(cfg["splits_dir"], "gtsdb_split.json")
    if not os.path.exists(split_path):
        print(f"ERROR: Split not found at {split_path}. Run make_splits.py first.")
        sys.exit(1)
    split = load_split(split_path)
    rows = parse_gtsdb_gt(os.path.join(cfg["gtsdb_root"], "gt.txt"))

    # Group annotations by image.
    by_image = defaultdict(list)
    for r in rows:
        by_image[r["img"]].append(r)

    gtsdb_root = cfg["gtsdb_root"]

    # ---- Step 1: Train HOG+SVM on training crops ----
    print("[BASELINE] Training HOG+SVM on training-set crops...")
    train_crops, train_labels = [], []
    for img_name in split["train"]:
        anns = by_image.get(img_name, [])
        if not anns:
            continue
        img = load_image(gtsdb_root, img_name)
        for r in anns:
            x1, y1, x2, y2 = r["box"]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            train_crops.append(cv2.resize(crop, (32, 32)))
            train_labels.append(r["cls"])

    if not train_crops:
        print("ERROR: No training crops found. Check data paths.")
        sys.exit(1)

    svm = train_hog_svm(train_crops, train_labels)
    print(f"  Trained on {len(train_crops)} crops ({len(set(train_labels))} classes)")

    # ---- Step 2: Evaluate on validation images ----
    print("[BASELINE] Evaluating on validation images...")
    det_recalls, det_precisions = [], []
    cls_correct, cls_total = 0, 0
    timings_ms = []

    for img_name in split["val"]:
        img = load_image(gtsdb_root, img_name)
        gt_anns = by_image.get(img_name, [])
        gt_boxes = [r["box"] for r in gt_anns]
        gt_labels = [r["cls"] for r in gt_anns]

        # Detection: colour + shape candidates.
        t0 = time.perf_counter()
        pred_boxes = color_shape_candidates(img)
        det_time = (time.perf_counter() - t0) * 1000

        if gt_boxes:
            rec, prec = detection_recall(pred_boxes, gt_boxes)
            det_recalls.append(rec)
            det_precisions.append(prec)

        # Classification on ground-truth crops (measures classifier in isolation).
        if gt_anns:
            gt_crops = extract_gt_crops(img, [r["box"] for r in gt_anns])
            if gt_crops:
                t1 = time.perf_counter()
                preds = predict_hog_svm(svm, gt_crops)
                cls_time = (time.perf_counter() - t1) * 1000
                det_time += cls_time
                cls_correct += int((preds == np.array(gt_labels[:len(preds)])).sum())
                cls_total += len(preds)

        timings_ms.append(det_time)

    # ---- Aggregate metrics ----
    avg_recall = float(np.mean(det_recalls)) if det_recalls else 0.0
    avg_precision = float(np.mean(det_precisions)) if det_precisions else 0.0
    cls_accuracy = cls_correct / cls_total if cls_total else 0.0
    avg_ms = float(np.mean(timings_ms)) if timings_ms else 0.0
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    metrics = {
        "detection": {
            "recall_at_iou50": round(avg_recall, 4),
            "precision_at_iou50": round(avg_precision, 4),
            "num_val_images": len(split["val"]),
            "num_val_images_with_signs": len(det_recalls),
        },
        "classification_on_gt_crops": {
            "accuracy": round(cls_accuracy, 4),
            "correct": cls_correct,
            "total": cls_total,
        },
        "speed": {
            "avg_ms_per_image": round(avg_ms, 2),
            "fps": round(fps, 1),
            "note": "CPU only, includes detection + classification on GT crops",
        },
    }

    print(f"\n--- Baseline Results (validation) ---")
    print(f"  Detection recall@IoU0.5:    {avg_recall:.3f}")
    print(f"  Detection precision@IoU0.5: {avg_precision:.3f}")
    print(f"  Classification accuracy:    {cls_accuracy:.3f} ({cls_correct}/{cls_total})")
    print(f"  Speed: {avg_ms:.1f} ms/image ({fps:.1f} FPS)")

    out_path = os.path.join(out_dir, "baseline_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[BASELINE] Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
