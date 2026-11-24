#!/usr/bin/env python3
"""Fine-tune the classifier on GTSDB training crops to close the domain gap.

Extracts sign crops from GTSDB training images using GT bounding boxes,
then fine-tunes the best RoadSignNet checkpoint on these crops mixed with
GTSRB validation-style transforms.

Usage:
    python scripts/finetune_on_gtsdb.py --config configs/kaggle.yaml \
        --classifier results/classification/scratch_aug1_lr0.001/best.pt \
        --epochs 15
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadsign.data import parse_gtsdb_gt, load_split, assert_not_test
from roadsign.model import RoadSignNet


class GTSDBCropDataset(Dataset):
    """Dataset of sign crops extracted from GTSDB images using GT boxes."""

    def __init__(self, gtsdb_root, gt_rows, split_images, transform=None):
        self.gtsdb_root = gtsdb_root
        self.transform = transform
        self.crops = []
        split_set = set(split_images)
        for r in gt_rows:
            if r["img"] in split_set:
                self.crops.append(r)

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        r = self.crops[idx]
        img_path = os.path.join(self.gtsdb_root, r["img"])
        assert_not_test(img_path)
        img = Image.open(img_path).convert("RGB")
        x1, y1, x2, y2 = r["box"]
        crop = img.crop((max(0, x1), max(0, y1), x2, y2))
        label = r["cls"]
        if self.transform:
            crop = self.transform(crop)
        return crop, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kaggle.yaml")
    parser.add_argument("--classifier", required=True, help="Path to best.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt_path = os.path.join(cfg["gtsdb_root"], "gt.txt")
    gt_rows = parse_gtsdb_gt(gt_path)

    split_path = os.path.join(cfg["splits_dir"], "gtsdb_split.json")
    split = load_split(split_path)

    train_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3401, 0.3120, 0.3214],
                             std=[0.2724, 0.2608, 0.2669]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3401, 0.3120, 0.3214],
                             std=[0.2724, 0.2608, 0.2669]),
    ])

    train_ds = GTSDBCropDataset(cfg["gtsdb_root"], gt_rows, split["train"], train_tf)
    val_ds = GTSDBCropDataset(cfg["gtsdb_root"], gt_rows, split["val"], val_tf)

    print(f"[FINETUNE] GTSDB crops: {len(train_ds)} train, {len(val_ds)} val")
    print(f"[FINETUNE] Device: {device}, lr={args.lr}, epochs={args.epochs}")

    if len(train_ds) == 0:
        print("ERROR: No training crops found")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = RoadSignNet(num_classes=43)
    model.load_state_dict(torch.load(args.classifier, map_location=device, weights_only=True))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    out_dir = os.path.join(cfg.get("output_dir", "results"), "classification", "finetuned_gtsdb")
    os.makedirs(out_dir, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_acc = correct / total
        train_loss = total_loss / total

        model.eval()
        v_correct, v_total, v_loss = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                v_loss += loss.item() * imgs.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total += imgs.size(0)

        val_acc = v_correct / v_total if v_total > 0 else 0
        val_loss = v_loss / v_total if v_total > 0 else 0
        scheduler.step(val_loss)

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            marker = " *"

        lr = optimizer.param_groups[0]["lr"]
        print(f"[{epoch:02d}/{args.epochs}] train {train_acc:.3f}/{train_loss:.4f}  "
              f"val {val_acc:.3f}/{val_loss:.4f}  lr={lr:.1e}{marker}")

    torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
    print(f"\n[FINETUNE] Best GTSDB val accuracy: {best_acc:.4f}")
    print(f"[FINETUNE] Saved to: {out_dir}")


if __name__ == "__main__":
    main()
