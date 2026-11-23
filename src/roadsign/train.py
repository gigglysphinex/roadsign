"""Training utilities for GTSRB classification (Phase 3, EF2).

Provides a Dataset class for GTSRB crops, training/evaluation loops,
and helpers for logging experiment results.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .data import load_split, assert_not_test


class GTSRBDataset(Dataset):
    """Loads GTSRB sign crops from a split JSON file."""

    def __init__(self, gtsrb_root, file_list, transform=None, cache=True):
        self.gtsrb_root = gtsrb_root
        self.files = file_list
        self.transform = transform
        train_root = os.path.join(gtsrb_root, "Train")
        self.base = train_root if os.path.isdir(train_root) else gtsrb_root
        self._cache = {}
        if cache:
            self._preload()

    def _preload(self):
        from PIL import Image
        print(f"[CACHE] Pre-loading {len(self.files)} images into memory...")
        for idx, rel in enumerate(self.files):
            path = os.path.join(self.base, rel)
            assert_not_test(path)
            img = Image.open(path).convert("RGB").resize((32, 32))
            label = int(rel.split("/")[0]) if "/" in rel else int(os.path.basename(rel).split("_")[0])
            self._cache[idx] = (img, label)
        print(f"[CACHE] Done.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx in self._cache:
            img, label = self._cache[idx]
        else:
            rel = self.files[idx]
            path = os.path.join(self.base, rel)
            assert_not_test(path)
            from PIL import Image
            img = Image.open(path).convert("RGB").resize((32, 32))
            label = int(rel.split("/")[0]) if "/" in rel else int(os.path.basename(rel).split("_")[0])
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label


def get_transforms(augment=False):
    """Return (train_transform, val_transform)."""
    normalize = transforms.Normalize(
        mean=[0.3401, 0.3120, 0.3214],
        std=[0.2724, 0.2608, 0.2669],
    )
    val_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize,
    ])
    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = val_tf
    return train_tf, val_tf


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, device,
                epochs=30, lr=1e-3, out_dir="results/classification"):
    """Full training loop. Returns history dict and best val accuracy."""
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_epoch = 0.0, 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_acc)
        elapsed = time.time() - t0

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(va_loss, 4))
        history["val_acc"].append(round(va_acc, 4))

        tag = ""
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            tag = " *"

        print(f"[{epoch:02d}/{epochs}] "
              f"train {tr_acc:.3f} / {tr_loss:.4f}  "
              f"val {va_acc:.3f} / {va_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.1e}  "
              f"{elapsed:.1f}s{tag}")

    torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history, best_acc, best_epoch
