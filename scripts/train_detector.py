#!/usr/bin/env python3
"""Phase 4: Train YOLOv8 detector on GTSDB.

Two modes:
  --classes 43  : end-to-end 43-class detection (detect + classify)
  --classes 1   : single-class "sign" detector (for detect-then-classify pipeline)

Requires: pip install ultralytics

Usage:
    python scripts/train_detector.py --config configs/kaggle.yaml --classes 1 --epochs 50
    python scripts/train_detector.py --config configs/kaggle.yaml --classes 43 --epochs 50
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 detector on GTSDB")
    parser.add_argument("--config", default="configs/kaggle.yaml")
    parser.add_argument("--classes", type=int, choices=[1, 43], default=1)
    parser.add_argument("--model-size", choices=["n", "s", "m"], default="n",
                        help="YOLOv8 variant: n=nano, s=small, m=medium")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Step 1: Export YOLO dataset if not already done
    yolo_root = cfg.get("yolo_dir", "results/yolo")
    variant = f"{args.classes}cls"
    data_yaml = os.path.join(yolo_root, variant, "data.yaml")

    if not os.path.exists(data_yaml):
        print(f"[YOLO] Exporting dataset ({variant})...")
        ret = os.system(
            f"{sys.executable} scripts/export_yolo.py --config {args.config} --copy"
        )
        if ret != 0:
            print("ERROR: export_yolo.py failed")
            sys.exit(1)

    if not os.path.exists(data_yaml):
        print(f"ERROR: {data_yaml} not found after export")
        sys.exit(1)

    print(f"[CONFIG] classes={args.classes}, model=yolov8{args.model_size}, "
          f"epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}")

    # Step 2: Train YOLOv8
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model_name = f"yolov8{args.model_size}.pt"
    model = YOLO(model_name)

    out_dir = os.path.join(
        cfg.get("output_dir", "results"), "detection",
        f"yolov8{args.model_size}_{variant}"
    )

    results = model.train(
        data=os.path.abspath(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=os.path.abspath(out_dir),
        name="train",
        exist_ok=True,
        seed=cfg.get("seed", 42),
        patience=10,
        save=True,
        plots=True,
    )

    # Step 3: Validate
    metrics = model.val(data=os.path.abspath(data_yaml))

    # Step 4: Save experiment log
    exp = {
        "task": "detection",
        "classes": args.classes,
        "model": f"yolov8{args.model_size}",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": cfg.get("seed", 42),
        "git_commit": get_git_hash(),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }

    exp_path = os.path.join(out_dir, "experiment.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(exp_path, "w") as f:
        json.dump(exp, f, indent=2)

    print(f"\n--- Detection Result ---")
    print(f"  mAP@50:    {exp['mAP50']:.4f}")
    print(f"  mAP@50:95: {exp['mAP50_95']:.4f}")
    print(f"  Precision:  {exp['precision']:.4f}")
    print(f"  Recall:     {exp['recall']:.4f}")
    print(f"  Saved to:   {out_dir}")


if __name__ == "__main__":
    main()
