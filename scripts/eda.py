#!/usr/bin/env python3
"""Exploratory Data Analysis (Phase 1, EF1).

Loads GTSDB and GTSRB training data, prints class distributions, box-size
statistics, and runs the annotation anomaly report.  Saves everything as JSON
so the numbers are available for the written report.

Usage:
    python scripts/eda.py --config configs/phase1.yaml
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadsign.data import (
    parse_gtsdb_gt, anomaly_report, class_counts, assert_not_test,
    NUM_CLASSES, GTSDB_SIZE,
)


def gtsdb_box_stats(rows):
    """Compute width, height, and area statistics for GTSDB bounding boxes."""
    widths, heights = [], []
    for r in rows:
        x1, y1, x2, y2 = r["box"]
        widths.append(x2 - x1)
        heights.append(y2 - y1)
    if not widths:
        return {}
    import numpy as np
    w = np.array(widths)
    h = np.array(heights)
    areas = w * h
    return {
        "count": len(w),
        "width":  {"min": int(w.min()), "max": int(w.max()), "mean": round(float(w.mean()), 1),
                    "median": round(float(np.median(w)), 1)},
        "height": {"min": int(h.min()), "max": int(h.max()), "mean": round(float(h.mean()), 1),
                    "median": round(float(np.median(h)), 1)},
        "area":   {"min": int(areas.min()), "max": int(areas.max()),
                    "mean": round(float(areas.mean()), 1)},
        # Size buckets useful for robustness analysis (ENF4).
        "small_leq_32px": int((w <= 32).sum()),
        "medium_33_96px": int(((w > 32) & (w <= 96)).sum()),
        "large_gt_96px":  int((w > 96).sum()),
    }


def gtsrb_class_counts(gtsrb_root):
    """Walk the GTSRB Train folder and count images per class."""
    train_root = os.path.join(gtsrb_root, "Train")
    if not os.path.isdir(train_root):
        # Some layouts have classes directly under gtsrb_root.
        train_root = gtsrb_root
    counts = defaultdict(int)
    total = 0
    for cls_dir in sorted(os.listdir(train_root)):
        cls_path = os.path.join(train_root, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        try:
            cls_id = int(cls_dir)
        except ValueError:
            continue
        n = sum(1 for f in os.listdir(cls_path) if f.lower().endswith((".ppm", ".png", ".jpg")))
        counts[cls_id] = n
        total += n
    return dict(sorted(counts.items())), total


def main():
    parser = argparse.ArgumentParser(description="EDA for GTSDB and GTSRB")
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["eda_dir"]
    os.makedirs(out_dir, exist_ok=True)

    report = {}

    # --- GTSDB ---
    gt_path = os.path.join(cfg["gtsdb_root"], "gt.txt")
    print(f"[EDA] Loading GTSDB ground truth from {gt_path}")
    rows = parse_gtsdb_gt(gt_path)

    # Guard: make sure we only touch training images.
    train_imgs = sorted({r["img"] for r in rows})
    for img in train_imgs:
        assert_not_test(img)

    # All images in the training folder (some may have no annotation).
    all_imgs = sorted(
        f for f in os.listdir(cfg["gtsdb_root"])
        if f.lower().endswith((".ppm", ".png", ".jpg")) and not is_gtsdb_test(f)
    )

    gtsdb_cc = class_counts(rows)
    counts_summary, detail = anomaly_report(rows, all_images=all_imgs)
    box_stats = gtsdb_box_stats(rows)

    report["gtsdb"] = {
        "num_training_images": len(all_imgs),
        "num_annotations": len(rows),
        "class_counts": gtsdb_cc,
        "box_stats": box_stats,
        "anomalies": counts_summary,
    }
    print(f"  Images: {len(all_imgs)}, Annotations: {len(rows)}")
    print(f"  Classes present: {len(gtsdb_cc)} / {NUM_CLASSES}")
    print(f"  Box sizes -- {box_stats.get('small_leq_32px', '?')} small, "
          f"{box_stats.get('medium_33_96px', '?')} medium, {box_stats.get('large_gt_96px', '?')} large")
    print(f"  Anomalies: {counts_summary if counts_summary else 'none'}")

    # --- GTSRB ---
    print(f"\n[EDA] Scanning GTSRB training data at {cfg['gtsrb_root']}")
    gtsrb_cc, gtsrb_total = gtsrb_class_counts(cfg["gtsrb_root"])
    report["gtsrb"] = {
        "num_training_images": gtsrb_total,
        "class_counts": gtsrb_cc,
    }
    print(f"  Total training images: {gtsrb_total}")
    print(f"  Classes present: {len(gtsrb_cc)} / {NUM_CLASSES}")
    if gtsrb_cc:
        min_cls = min(gtsrb_cc, key=gtsrb_cc.get)
        max_cls = max(gtsrb_cc, key=gtsrb_cc.get)
        print(f"  Smallest class: {min_cls} ({gtsrb_cc[min_cls]} images)")
        print(f"  Largest class:  {max_cls} ({gtsrb_cc[max_cls]} images)")

    # --- Save ---
    out_path = os.path.join(out_dir, "eda_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[EDA] Report saved to {out_path}")


def is_gtsdb_test(name):
    """Quick check used only in EDA to filter the directory listing."""
    from roadsign.data import is_gtsdb_test_image
    return is_gtsdb_test_image(name)


if __name__ == "__main__":
    main()
