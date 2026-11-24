#!/usr/bin/env python3
"""Phase 3: Train GTSRB classifiers (scratch CNN vs pretrained ResNet-18).

Runs one experiment at a time, controlled by --model and --augment flags.
Logs config, seed, git commit, and all metrics to JSON.

Usage:
    python scripts/train_classifier.py --config configs/phase1.yaml --model scratch --augment
    python scripts/train_classifier.py --config configs/phase1.yaml --model resnet18 --augment
    python scripts/train_classifier.py --config configs/phase1.yaml --model scratch --no-augment
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadsign.data import load_split
from roadsign.model import RoadSignNet
from roadsign.train import (
    GTSRBDataset, get_transforms, train_model,
)


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def build_resnet18(num_classes=43):
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train GTSRB classifier")
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--model", choices=["scratch", "resnet18"], default="scratch")
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable image pre-loading into RAM")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CONFIG] model={args.model}, augment={args.augment}, lr={args.lr}, "
          f"epochs={args.epochs}, bs={args.batch_size}, device={device}")

    # Load split
    split_path = os.path.join(cfg["splits_dir"], "gtsrb_split.json")
    if not os.path.exists(split_path):
        print(f"ERROR: Split not found at {split_path}. Run make_splits.py first.")
        sys.exit(1)
    split = load_split(split_path)
    print(f"[DATA] train={len(split['train'])}, val={len(split['val'])}")

    # Transforms
    train_tf, val_tf = get_transforms(augment=args.augment)

    # Datasets and loaders
    use_cache = not args.no_cache
    train_ds = GTSRBDataset(cfg["gtsrb_root"], split["train"], transform=train_tf, cache=use_cache)
    val_ds = GTSRBDataset(cfg["gtsrb_root"], split["val"], transform=val_tf, cache=use_cache)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # Model
    if args.model == "scratch":
        model = RoadSignNet(num_classes=43)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[MODEL] RoadSignNet from scratch, {n_params:,} params")
    else:
        model = build_resnet18(num_classes=43)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[MODEL] ResNet-18 pretrained, {n_params:,} params")

    # Output directory
    tag = f"{args.model}_aug{int(args.augment)}_lr{args.lr}"
    out_dir = os.path.join(cfg.get("output_dir", "results"), "classification", tag)

    # Train
    history, best_acc, best_epoch = train_model(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, out_dir=out_dir
    )

    # Save experiment config
    exp = {
        "model": args.model,
        "augment": args.augment,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": seed,
        "git_commit": get_git_hash(),
        "device": str(device),
        "n_params": n_params,
        "best_val_acc": round(best_acc, 4),
        "best_epoch": best_epoch,
        "train_size": len(split["train"]),
        "val_size": len(split["val"]),
        "torch_version": torch.__version__,
    }
    with open(os.path.join(out_dir, "experiment.json"), "w") as f:
        json.dump(exp, f, indent=2)

    print(f"\n--- Result ---")
    print(f"  Best val accuracy: {best_acc:.4f} (epoch {best_epoch})")
    print(f"  Parameters: {n_params:,}")
    print(f"  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
