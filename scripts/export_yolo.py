#!/usr/bin/env python3
"""Export GTSDB annotations to Ultralytics YOLO directory layout (Phase 1, EF1/EF3).

Creates two variants:
  results/yolo/43cls/   -- 43-class detection (EF3)
  results/yolo/1cls/    -- single-class "sign" detection (for A1 detect-then-classify)

Each has: images/train, images/val, labels/train, labels/val, data.yaml.
Images are symlinked (not copied) to save disk space.

Usage:
    python scripts/export_yolo.py --config configs/phase1.yaml
"""
import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadsign.data import (
    parse_gtsdb_gt, load_split, to_yolo_lines, assert_not_test,
    NUM_CLASSES, GTSDB_SIZE,
)


# GTSDB class names (German Traffic Sign Detection Benchmark).
# Grouped: 0-9 speed limits, 10 no overtaking, ... 42 end of no overtaking.
GTSDB_NAMES = {
    0: "speed_20", 1: "speed_30", 2: "speed_50", 3: "speed_60", 4: "speed_70",
    5: "speed_80", 6: "end_speed_80", 7: "speed_100", 8: "speed_120",
    9: "no_overtaking", 10: "no_overtaking_trucks",
    11: "priority_next", 12: "priority_road", 13: "yield", 14: "stop",
    15: "no_vehicles", 16: "no_trucks", 17: "no_entry", 18: "danger",
    19: "curve_left", 20: "curve_right", 21: "double_curve", 22: "bumpy",
    23: "slippery", 24: "narrows_right", 25: "roadwork", 26: "traffic_signals",
    27: "pedestrians", 28: "children", 29: "bicycles", 30: "ice",
    31: "wild_animals", 32: "end_all", 33: "right_only", 34: "left_only",
    35: "straight_only", 36: "straight_right", 37: "straight_left",
    38: "keep_right", 39: "keep_left", 40: "roundabout", 41: "end_no_overtaking",
    42: "end_no_overtaking_trucks",
}


def write_yolo_dataset(gtsdb_root, rows, split, out_root, single_class=False):
    """Write images (as symlinks) and label .txt files in YOLO layout."""
    img_w, img_h = GTSDB_SIZE

    # Group annotations by image name.
    by_image = defaultdict(list)
    for r in rows:
        by_image[r["img"]].append(r)

    for subset in ("train", "val"):
        img_dir = os.path.join(out_root, "images", subset)
        lbl_dir = os.path.join(out_root, "labels", subset)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for img_name in split[subset]:
            assert_not_test(img_name)
            src = os.path.abspath(os.path.join(gtsdb_root, img_name))
            dst = os.path.join(img_dir, img_name)
            # Symlink the image to avoid copying large files.
            if not os.path.exists(dst):
                os.symlink(src, dst)

            stem = os.path.splitext(img_name)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            lines = to_yolo_lines(by_image.get(img_name, []), img_w, img_h,
                                  single_class=single_class)
            with open(lbl_path, "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

    # Write data.yaml for Ultralytics.
    nc = 1 if single_class else NUM_CLASSES
    names = {0: "sign"} if single_class else GTSDB_NAMES
    data_yaml = {
        "path": os.path.abspath(out_root),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": names,
    }
    yaml_path = os.path.join(out_root, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Export GTSDB to YOLO format")
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    split_path = os.path.join(cfg["splits_dir"], "gtsdb_split.json")
    if not os.path.exists(split_path):
        print(f"ERROR: Split file not found at {split_path}. Run make_splits.py first.")
        sys.exit(1)
    split = load_split(split_path)

    gt_path = os.path.join(cfg["gtsdb_root"], "gt.txt")
    rows = parse_gtsdb_gt(gt_path)

    yolo_root = cfg["yolo_dir"]

    # 43-class variant.
    out_43 = os.path.join(yolo_root, "43cls")
    p = write_yolo_dataset(cfg["gtsdb_root"], rows, split, out_43, single_class=False)
    n_train = len(split["train"])
    n_val = len(split["val"])
    print(f"[YOLO] 43-class: {n_train} train / {n_val} val images -> {p}")

    # 1-class variant (for A1 detect-then-classify).
    out_1 = os.path.join(yolo_root, "1cls")
    p = write_yolo_dataset(cfg["gtsdb_root"], rows, split, out_1, single_class=True)
    print(f"[YOLO]  1-class: {n_train} train / {n_val} val images -> {p}")


if __name__ == "__main__":
    main()
