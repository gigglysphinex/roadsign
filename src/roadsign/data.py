"""Data pipeline (EF1): parsing, annotation quality report, leak-free splits, label conversion.

GTSDB  gt.txt line:  00000.ppm;x1;y1;x2;y2;classId     (detection, 43 classes, ids 0-42)
GTSRB  train files:  <class>/<track>_<frame>.ppm        (classification, 30 frames per physical sign)
"""
import json
import os
import random
import re
from collections import defaultdict

from .boxes import xyxy_to_yolo

NUM_CLASSES = 43
GTSDB_SIZE = (1360, 800)  # (width, height) of GTSDB images

# --- Test-set guard -----------------------------------------------------------
# GTSDB test images are 00600.ppm through 00899.ppm.
# GTSRB official test set lives in a folder usually named "Test" or "Final_Test".
# This guard must be called before any data-loading function touches a path.

_GTSDB_TEST_RE = re.compile(r"0{0,2}(6\d{2}|7\d{2}|8\d{2}|899)")  # 600..899


def is_gtsdb_test_image(name: str) -> bool:
    """True if `name` matches a GTSDB test image (index 600-899)."""
    stem = os.path.splitext(os.path.basename(name))[0]
    try:
        idx = int(stem)
    except ValueError:
        return False
    return 600 <= idx <= 899


def assert_not_test(path: str) -> None:
    """Raise ValueError if `path` points to a GTSDB test image or GTSRB official test set.

    Call this in every data-loading / export function to prevent accidental leakage.
    """
    base = os.path.basename(path)
    if is_gtsdb_test_image(base):
        raise ValueError(f"FORBIDDEN: {path} is a GTSDB test image (00600-00899).")
    # GTSRB official test folders are typically named "Test" or "Final_Test"
    parts = path.replace("\\", "/").lower().split("/")
    for forbidden in ("final_test", "gt_final_test"):
        if forbidden in parts:
            raise ValueError(f"FORBIDDEN: {path} belongs to the GTSRB official test set.")


def parse_gtsdb_gt(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, x1, y1, x2, y2, c = line.split(";")
            rows.append(dict(img=name, box=(int(x1), int(y1), int(x2), int(y2)), cls=int(c)))
    return rows


def anomaly_report(rows, all_images=None, size_of=None):
    """Annotation quality report. size_of(img_name) -> (w, h); defaults to GTSDB_SIZE."""
    size_of = size_of or (lambda _n: GTSDB_SIZE)
    rep = defaultdict(list)
    seen = set()
    for r in rows:
        x1, y1, x2, y2 = r["box"]
        w, h = size_of(r["img"])
        if x2 <= x1 or y2 <= y1:
            rep["degenerate_box"].append(r)
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            rep["box_outside_image"].append(r)
        if not 0 <= r["cls"] < NUM_CLASSES:
            rep["class_out_of_range"].append(r)
        key = (r["img"], r["box"], r["cls"])
        if key in seen:
            rep["duplicate"].append(r)
        seen.add(key)
    if all_images is not None:
        annotated = {r["img"] for r in rows}
        rep["image_without_annotation"] = sorted(set(all_images) - annotated)
    return {k: len(v) for k, v in rep.items()}, rep


def class_counts(rows):
    counts = defaultdict(int)
    for r in rows:
        counts[r["cls"]] += 1
    return dict(sorted(counts.items()))


def block_split(image_names, val_frac=0.2, block=20, seed=0):
    """Split images into train/val by contiguous blocks of `block` consecutive images.

    Consecutive GTSDB frames come from the same drive and look alike, so a random
    image-level split would leak near-duplicates into validation.
    """
    names = sorted(image_names)
    blocks = [names[i:i + block] for i in range(0, len(names), block)]
    rng = random.Random(seed)
    rng.shuffle(blocks)
    n_val = max(1, round(len(blocks) * val_frac))
    val = sorted(n for b in blocks[:n_val] for n in b)
    train = sorted(n for b in blocks[n_val:] for n in b)
    return train, val


def gtsrb_track_split(files, val_frac=0.2, seed=0):
    """files: paths like '00012/00005_00017.ppm'. Split by (class, track) so that the 30
    frames of one physical sign are never spread over train and validation."""
    tracks = defaultdict(list)
    for f in files:
        cls_dir, fname = f.replace("\\", "/").split("/")[-2:]
        tracks[(cls_dir, fname.split("_")[0])].append(f)
    by_class = defaultdict(list)
    for (cls_dir, tr) in tracks:
        by_class[cls_dir].append((cls_dir, tr))
    rng = random.Random(seed)
    train, val = [], []
    for cls_dir, keys in sorted(by_class.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * val_frac)) if len(keys) > 1 else 0
        for i, k in enumerate(keys):
            (val if i < n_val else train).extend(tracks[k])
    return sorted(train), sorted(val)


def save_split(path, **splits):
    with open(path, "w") as f:
        json.dump({k: list(v) for k, v in splits.items()}, f, indent=1)


def load_split(path):
    with open(path) as f:
        return json.load(f)


def to_yolo_lines(rows_of_image, img_w, img_h, single_class=False):
    """YOLO label lines for one image. single_class=True gives a class-agnostic 'sign'
    detector (class 0), used by the detect-then-classify pipeline."""
    lines = []
    for r in rows_of_image:
        cx, cy, w, h = xyxy_to_yolo(r["box"], img_w, img_h)
        c = 0 if single_class else r["cls"]
        lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines
