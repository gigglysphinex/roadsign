#!/usr/bin/env python3
"""Create and save train/val splits for GTSDB and GTSRB (Phase 1, ENF6).

Splits are deterministic (fixed seed) and saved as JSON.  Once generated they
are committed to the repo and reloaded -- never recomputed.

Usage:
    python scripts/make_splits.py --config configs/phase1.yaml
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadsign.data import (
    block_split, gtsrb_track_split, save_split, assert_not_test,
    is_gtsdb_test_image,
)


def collect_gtsdb_train_images(gtsdb_root):
    """List GTSDB training image filenames (00000-00599), excluding the test set."""
    names = []
    for f in sorted(os.listdir(gtsdb_root)):
        if not f.lower().endswith((".ppm", ".png", ".jpg")):
            continue
        if is_gtsdb_test_image(f):
            continue
        names.append(f)
    return names


def collect_gtsrb_train_files(gtsrb_root):
    """List GTSRB training file relative paths like '00012/00005_00017.ppm'."""
    train_root = os.path.join(gtsrb_root, "Train")
    if not os.path.isdir(train_root):
        train_root = gtsrb_root

    files = []
    for cls_dir in sorted(os.listdir(train_root)):
        cls_path = os.path.join(train_root, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        # Guard: skip the official test folder.
        try:
            assert_not_test(cls_path)
        except ValueError:
            continue
        for fname in sorted(os.listdir(cls_path)):
            if fname.lower().endswith((".ppm", ".png", ".jpg")):
                files.append(f"{cls_dir}/{fname}")
    return files


def main():
    parser = argparse.ArgumentParser(description="Generate and save train/val splits")
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["splits_dir"]
    os.makedirs(out_dir, exist_ok=True)
    seed = cfg["seed"]

    # --- GTSDB split (by blocks of consecutive images) ---
    gtsdb_imgs = collect_gtsdb_train_images(cfg["gtsdb_root"])
    print(f"[SPLIT] GTSDB training images: {len(gtsdb_imgs)}")
    gtsdb_train, gtsdb_val = block_split(
        gtsdb_imgs,
        val_frac=cfg["gtsdb_val_frac"],
        block=cfg["gtsdb_block_size"],
        seed=seed,
    )
    gtsdb_path = os.path.join(out_dir, "gtsdb_split.json")
    save_split(gtsdb_path, train=gtsdb_train, val=gtsdb_val)
    print(f"  train: {len(gtsdb_train)}, val: {len(gtsdb_val)}")
    print(f"  Saved to {gtsdb_path}")

    # --- GTSRB split (by physical sign track) ---
    gtsrb_files = collect_gtsrb_train_files(cfg["gtsrb_root"])
    print(f"\n[SPLIT] GTSRB training files: {len(gtsrb_files)}")
    gtsrb_train, gtsrb_val = gtsrb_track_split(
        gtsrb_files,
        val_frac=cfg["gtsrb_val_frac"],
        seed=seed,
    )
    gtsrb_path = os.path.join(out_dir, "gtsrb_split.json")
    save_split(gtsrb_path, train=gtsrb_train, val=gtsrb_val)
    print(f"  train: {len(gtsrb_train)}, val: {len(gtsrb_val)}")
    print(f"  Saved to {gtsrb_path}")


if __name__ == "__main__":
    main()
