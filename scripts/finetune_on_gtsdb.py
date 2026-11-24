#!/usr/bin/env python3
"""Fine-tune the classifier on GTSDB DETECTOR crops to close the domain gap.

Runs the trained YOLO detector on GTSDB training images, matches each
detection to the nearest GT box (IoU >= 0.5), and uses the DETECTOR's
crop with the GT label. This teaches the classifier to handle the exact
crop style the detector produces at inference time.

Usage:
    python scripts/finetune_on_gtsdb.py --config configs/kaggle.yaml \
        --classifier results/classification/scratch_aug1_lr0.001/best.pt \
        --detector results/detection/yolov8n_1cls/train/weights/best.pt \
        --epochs 20
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadsign.data import parse_gtsdb_gt, load_split, assert_not_test
from roadsign.boxes import iou_matrix
from roadsign.model import RoadSignNet


def extract_detector_crops(detector_weights, gtsdb_root, gt_rows, split_images):
    """Run detector on training images and match crops to GT labels."""
    from ultralytics import YOLO

    detector = YOLO(detector_weights)
    split_set = set(split_images)

    gt_by_image = {}
    for r in gt_rows:
        if r["img"] in split_set:
            gt_by_image.setdefault(r["img"], []).append(r)

    crops = []
    for img_name in sorted(gt_by_image.keys()):
        img_path = os.path.join(gtsdb_root, img_name)
        assert_not_test(img_path)
        img = Image.open(img_path).convert("RGB")

        results = detector(img, conf=0.15, iou=0.45, verbose=False)

        pred_boxes = []
        for r in results:
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    box = r.boxes.xyxy[i].cpu().numpy().astype(int)
                    pred_boxes.append(box)

        if not pred_boxes:
            continue

        gt_list = gt_by_image[img_name]
        gt_arr = np.array([g["box"] for g in gt_list])
        pred_arr = np.array(pred_boxes)
        ious = iou_matrix(pred_arr, gt_arr)

        matched_gt = set()
        for pi in range(len(pred_boxes)):
            best_gi = ious[pi].argmax()
            if ious[pi, best_gi] >= 0.5 and best_gi not in matched_gt:
                matched_gt.add(best_gi)
                x1, y1, x2, y2 = pred_boxes[pi]
                crop = img.crop((max(0, x1), max(0, y1), x2, y2))
                label = gt_list[best_gi]["cls"]
                crops.append((crop, label))

    return crops


class CropListDataset(Dataset):
    """Dataset from a list of (PIL crop, label) pairs."""

    def __init__(self, crops, transform=None):
        self.crops = crops
        self.transform = transform

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop, label = self.crops[idx]
        if self.transform:
            crop = self.transform(crop)
        return crop, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kaggle.yaml")
    parser.add_argument("--classifier", required=True, help="Path to classifier best.pt")
    parser.add_argument("--detector", required=True, help="Path to YOLO 1-class best.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt_path = os.path.join(cfg["gtsdb_root"], "gt.txt")
    gt_rows = parse_gtsdb_gt(gt_path)

    split_path = os.path.join(cfg["splits_dir"], "gtsdb_split.json")
    split = load_split(split_path)

    print("[FINETUNE] Extracting detector crops from GTSDB training images...")
    train_crops = extract_detector_crops(
        args.detector, cfg["gtsdb_root"], gt_rows, split["train"])
    val_crops = extract_detector_crops(
        args.detector, cfg["gtsdb_root"], gt_rows, split["val"])

    print(f"[FINETUNE] Detector crops: {len(train_crops)} train, {len(val_crops)} val")

    if len(train_crops) == 0:
        print("ERROR: No training crops extracted")
        sys.exit(1)

    train_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
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

    train_ds = CropListDataset(train_crops, train_tf)
    val_ds = CropListDataset(val_crops, val_tf)

    print(f"[FINETUNE] Device: {device}, lr={args.lr}, epochs={args.epochs}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = RoadSignNet(num_classes=43)
    model.load_state_dict(torch.load(args.classifier, map_location=device, weights_only=True))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
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
        scheduler.step()

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            marker = " *"

        lr = optimizer.param_groups[0]["lr"]
        print(f"[{epoch:02d}/{args.epochs}] train {train_acc:.3f}/{train_loss:.4f}  "
              f"val {val_acc:.3f}/{val_loss:.4f}  lr={lr:.1e}{marker}")

    torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
    print(f"\n[FINETUNE] Best detector-crop val accuracy: {best_acc:.4f}")
    print(f"[FINETUNE] Saved to: {out_dir}")


if __name__ == "__main__":
    main()
