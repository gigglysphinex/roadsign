#!/usr/bin/env python3
"""Prepare Kaggle environment: convert YOLO-format GTSDB labels to gt.txt
and set up directory structure so all other scripts work unchanged.

The Kaggle GTSDB dataset has:
  GTSDB_Train_and_Test/Train/images/00000.jpg ...
  GTSDB_Train_and_Test/Train/labels/00000.txt ...  (YOLO format)
  GTSDB_Train_and_Test/Test/images/...

This script:
  1. Reads the YOLO labels and converts them to the original gt.txt format.
  2. Creates a working directory with symlinks so our scripts see
     a flat folder of images + gt.txt.

Usage (in Kaggle notebook):
    python scripts/prepare_kaggle.py
"""
import os
import sys
from pathlib import Path

# GTSDB image dimensions (all images are 1360 x 800).
IMG_W, IMG_H = 1360, 800

# --- Kaggle paths (adjust if the dataset slug changes) ---
GTSDB_INPUT = "/kaggle/input/datasets/icebearogo/german-traffic-sign-detection-gtsdb-dataset/GTSDB_Train_and_Test"
GTSRB_INPUT = "/kaggle/input/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"

# Where we build our working copy (writable).
WORK = "/kaggle/working/roadsign"
GTSDB_WORK = os.path.join(WORK, "data", "gtsdb")
GTSRB_WORK = os.path.join(WORK, "data", "gtsrb")


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert normalised YOLO (cx, cy, w, h) to pixel (x1, y1, x2, y2)."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def convert_yolo_to_gt(labels_dir, images_dir, out_gt_path, append=False):
    """Read YOLO label files and produce a gt.txt in GTSDB format."""
    lines = []
    for lbl_file in sorted(os.listdir(labels_dir)):
        if not lbl_file.endswith(".txt"):
            continue
        stem = os.path.splitext(lbl_file)[0]
        # Find the matching image (could be .jpg or .ppm).
        img_name = None
        for ext in (".jpg", ".ppm", ".png"):
            candidate = stem + ext
            if os.path.exists(os.path.join(images_dir, candidate)):
                img_name = candidate
                break
        if img_name is None:
            continue

        with open(os.path.join(labels_dir, lbl_file)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, IMG_W, IMG_H)
                lines.append(f"{img_name};{x1};{y1};{x2};{y2};{cls}")

    mode = "a" if append else "w"
    with open(out_gt_path, mode) as f:
        f.write("\n".join(sorted(lines)) + "\n")
    return len(lines)


def setup_gtsdb():
    """Create a working GTSDB directory with images (symlinked) and gt.txt."""
    train_images = os.path.join(GTSDB_INPUT, "Train", "images")
    train_labels = os.path.join(GTSDB_INPUT, "Train", "labels")

    if not os.path.isdir(train_images):
        print(f"ERROR: GTSDB train images not found at {train_images}")
        sys.exit(1)

    os.makedirs(GTSDB_WORK, exist_ok=True)

    # Symlink each training image into the flat working directory.
    count = 0
    for img in sorted(os.listdir(train_images)):
        src = os.path.join(train_images, img)
        dst = os.path.join(GTSDB_WORK, img)
        if not os.path.exists(dst):
            os.symlink(src, dst)
        count += 1

    # Convert YOLO labels to gt.txt.
    gt_path = os.path.join(GTSDB_WORK, "gt.txt")
    n_annots = convert_yolo_to_gt(train_labels, train_images, gt_path)
    print(f"[PREP] GTSDB: {count} training images symlinked, {n_annots} annotations -> {gt_path}")

    # Also link test images (for final evaluation).
    test_images = os.path.join(GTSDB_INPUT, "Test", "images")
    test_labels = os.path.join(GTSDB_INPUT, "Test", "labels")
    if os.path.isdir(test_images):
        t_count = 0
        for img in sorted(os.listdir(test_images)):
            src = os.path.join(test_images, img)
            dst = os.path.join(GTSDB_WORK, img)
            if not os.path.exists(dst):
                os.symlink(src, dst)
            t_count += 1
        # Append test annotations to gt.txt.
        if os.path.isdir(test_labels):
            n_test = convert_yolo_to_gt(test_labels, test_images, gt_path,
                                         append=True)
            print(f"[PREP] GTSDB: {t_count} test images + {n_test} test annotations linked")
        else:
            print(f"[PREP] GTSDB: {t_count} test images linked (no labels)")
    else:
        print("[PREP] GTSDB: test images not found (final eval will skip)")


def setup_gtsrb():
    """Symlink the GTSRB Train folder so our scripts can find it."""
    train_src = os.path.join(GTSRB_INPUT, "Train")
    if not os.path.isdir(train_src):
        print(f"ERROR: GTSRB Train folder not found at {train_src}")
        sys.exit(1)

    os.makedirs(GTSRB_WORK, exist_ok=True)

    # Symlink the Train directory.
    dst = os.path.join(GTSRB_WORK, "Train")
    if not os.path.exists(dst):
        os.symlink(train_src, dst)

    # Count images.
    total = 0
    for cls_dir in sorted(os.listdir(train_src)):
        cls_path = os.path.join(train_src, cls_dir)
        if os.path.isdir(cls_path):
            total += sum(1 for f in os.listdir(cls_path)
                         if f.lower().endswith((".ppm", ".png", ".jpg")))
    print(f"[PREP] GTSRB: {total} training images linked at {dst}")

    # Symlink the Test directory (for final evaluation).
    for name in ("Test", "Final_Test"):
        test_src = os.path.join(GTSRB_INPUT, name)
        if os.path.isdir(test_src):
            test_dst = os.path.join(GTSRB_WORK, name)
            if not os.path.exists(test_dst):
                os.symlink(test_src, test_dst)
            print(f"[PREP] GTSRB: test set linked at {test_dst}")
            break
    else:
        print("[PREP] GTSRB: test set not found (final eval will skip classifier)")


def main():
    print("=" * 60)
    print("Preparing Kaggle data for RoadSign")
    print("=" * 60)
    setup_gtsdb()
    setup_gtsrb()
    print("\n[PREP] Done. Use configs/kaggle.yaml to run scripts.")


if __name__ == "__main__":
    main()
