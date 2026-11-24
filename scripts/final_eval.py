#!/usr/bin/env python3
"""Phase 5: One-shot final evaluation on the FROZEN test set.

This script runs ONCE. It evaluates:
  1. Classification: best RoadSignNet on the GTSRB official test set.
  2. Detection (1-class): YOLOv8n on GTSDB test images (00600-00899).
  3. Detection (43-class): YOLOv8n on GTSDB test images.
  4. Pipeline (A1): detect-then-classify on GTSDB test images.

Outputs a JSON report with all metrics.

Usage:
    python scripts/final_eval.py --config configs/kaggle.yaml \
        --classifier-weights results/classification/scratch_aug1_lr0.001/best.pt \
        --detector-1cls results/detection/yolov8n_1cls/train/weights/best.pt \
        --detector-43cls results/detection/yolov8n_43cls/train/weights/best.pt
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadsign.data import parse_gtsdb_gt, GTSDB_SIZE, NUM_CLASSES
from roadsign.boxes import iou_matrix


def eval_classifier(weights_path, gtsrb_root, device):
    """Evaluate classifier on the GTSRB official test set."""
    from torchvision import transforms
    from roadsign.model import RoadSignNet

    test_dir = None
    for name in ("Test", "Final_Test"):
        candidate = os.path.join(gtsrb_root, name)
        if os.path.isdir(candidate):
            test_dir = candidate
            break
    if test_dir is None:
        print("[WARN] GTSRB official test set not found. Skipping classifier eval.")
        return None

    csv_path = os.path.join(test_dir, "GT-final_test.csv")
    if not os.path.exists(csv_path):
        for f in os.listdir(test_dir):
            if f.lower().endswith(".csv"):
                csv_path = os.path.join(test_dir, f)
                break

    if not os.path.exists(csv_path):
        print(f"[WARN] Test CSV not found in {test_dir}. Skipping.")
        return None

    model = RoadSignNet(num_classes=43)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3401, 0.3120, 0.3214],
                             std=[0.2724, 0.2608, 0.2669]),
    ])

    from PIL import Image
    import csv

    correct, total = 0, 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        first_line = f.readline()
        print(f"[DEBUG] CSV header: {repr(first_line[:200])}")
        sep = ";" if ";" in first_line else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=sep)
        rows_list = list(reader)
        print(f"[DEBUG] CSV rows: {len(rows_list)}")
        if rows_list:
            print(f"[DEBUG] Columns: {list(rows_list[0].keys())}")
            print(f"[DEBUG] Row 0: {dict(rows_list[0])}")
        skipped = 0
        for row in rows_list:
            fname = row.get("Filename") or row.get("filename") or row.get("Path")
            if fname is None:
                if skipped == 0:
                    print(f"[DEBUG] No filename column found in row: {dict(row)}")
                skipped += 1
                continue
            label_str = row.get("ClassId") or row.get("classid") or row.get("ClassID") or row.get("Label")
            if label_str is None:
                if skipped == 0:
                    print(f"[DEBUG] No class column found in row: {dict(row)}")
                skipped += 1
                continue
            label = int(label_str)
            basename = os.path.basename(fname)
            stem = os.path.splitext(basename)[0]
            img_path = os.path.join(test_dir, basename)
            if not os.path.exists(img_path):
                for ext in (".png", ".jpg", ".ppm"):
                    alt = os.path.join(test_dir, stem + ext)
                    if os.path.exists(alt):
                        img_path = alt
                        break
                else:
                    if skipped < 3:
                        print(f"[DEBUG] Image not found: {basename} (tried {stem}.png/.jpg/.ppm)")
                    skipped += 1
                    continue

            img = Image.open(img_path).convert("RGB")
            inp = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(inp).argmax(1).item()
            if pred == label:
                correct += 1
                per_class_correct[label] += 1
            total += 1
            per_class_total[label] += 1

    acc = correct / total if total > 0 else 0
    print(f"[CLASSIFY] Test accuracy: {acc:.4f} ({correct}/{total})")
    return {"accuracy": round(acc, 4), "correct": correct, "total": total}


def eval_detector(weights_path, gtsdb_root, gt_rows, num_classes, device):
    """Evaluate YOLOv8 detector on GTSDB test images."""
    from ultralytics import YOLO

    test_images = sorted([
        f for f in os.listdir(gtsdb_root)
        if f.lower().endswith((".ppm", ".jpg", ".png"))
        and 600 <= int(os.path.splitext(f)[0]) <= 899
    ])

    if not test_images:
        print("[WARN] No GTSDB test images found. Skipping detection eval.")
        return None

    gt_by_image = defaultdict(list)
    for r in gt_rows:
        gt_by_image[r["img"]].append(r)

    model = YOLO(weights_path)
    single_class = (num_classes == 1)

    all_tp, all_fp, all_fn = 0, 0, 0
    all_ious = []

    for img_name in test_images:
        img_path = os.path.join(gtsdb_root, img_name)
        results = model(img_path, conf=0.25, iou=0.45, verbose=False)

        pred_boxes = []
        for r in results:
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    box = r.boxes.xyxy[i].cpu().numpy()
                    conf = float(r.boxes.conf[i].cpu())
                    cls = int(r.boxes.cls[i].cpu()) if not single_class else 0
                    pred_boxes.append({"box": box, "conf": conf, "cls": cls})

        gt_list = gt_by_image.get(img_name, [])
        gt_boxes = np.array([g["box"] for g in gt_list]) if gt_list else np.empty((0, 4))

        if len(pred_boxes) == 0:
            all_fn += len(gt_list)
            continue
        if len(gt_list) == 0:
            all_fp += len(pred_boxes)
            continue

        pred_arr = np.array([p["box"] for p in pred_boxes])
        ious = iou_matrix(pred_arr, gt_boxes)

        matched_gt = set()
        for pi in range(len(pred_boxes)):
            best_gi = ious[pi].argmax()
            if ious[pi, best_gi] >= 0.5 and best_gi not in matched_gt:
                all_tp += 1
                all_ious.append(ious[pi, best_gi])
                matched_gt.add(best_gi)
            else:
                all_fp += 1
        all_fn += len(gt_list) - len(matched_gt)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = float(np.mean(all_ious)) if all_ious else 0

    tag = "1-class" if single_class else "43-class"
    print(f"[DETECT {tag}] P={precision:.4f} R={recall:.4f} F1={f1:.4f} "
          f"mean_IoU={mean_iou:.4f} (TP={all_tp} FP={all_fp} FN={all_fn})")
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "tp": all_tp, "fp": all_fp, "fn": all_fn,
        "test_images": len(test_images),
    }


def eval_pipeline(detector_weights, classifier_weights, gtsdb_root, gt_rows, device):
    """Evaluate the A1 detect-then-classify pipeline on GTSDB test images."""
    from roadsign.pipeline import DetectThenClassify

    test_images = sorted([
        f for f in os.listdir(gtsdb_root)
        if f.lower().endswith((".ppm", ".jpg", ".png"))
        and 600 <= int(os.path.splitext(f)[0]) <= 899
    ])

    if not test_images:
        print("[WARN] No GTSDB test images found. Skipping pipeline eval.")
        return None

    gt_by_image = defaultdict(list)
    for r in gt_rows:
        gt_by_image[r["img"]].append(r)

    pipe = DetectThenClassify(
        detector_weights, classifier_weights,
        device=str(device), det_conf=0.25, nms_iou=0.45
    )

    tp_det, fp_det, fn_det = 0, 0, 0
    cls_correct, cls_total = 0, 0
    all_pairs = []

    for img_name in test_images:
        img_path = os.path.join(gtsdb_root, img_name)
        dets = pipe.predict(img_path)

        gt_list = gt_by_image.get(img_name, [])
        gt_boxes = np.array([g["box"] for g in gt_list]) if gt_list else np.empty((0, 4))

        if len(dets) == 0:
            fn_det += len(gt_list)
            continue
        if len(gt_list) == 0:
            fp_det += len(dets)
            continue

        pred_arr = np.array([d["box"] for d in dets])
        ious = iou_matrix(pred_arr, gt_boxes)

        matched_gt = set()
        for pi in range(len(dets)):
            best_gi = ious[pi].argmax()
            if ious[pi, best_gi] >= 0.5 and best_gi not in matched_gt:
                tp_det += 1
                matched_gt.add(best_gi)
                pred_cls = dets[pi]["class_id"]
                gt_cls = gt_list[best_gi]["cls"]
                all_pairs.append((pred_cls, gt_cls))
                if pred_cls == gt_cls:
                    cls_correct += 1
                cls_total += 1
            else:
                fp_det += 1
        fn_det += len(gt_list) - len(matched_gt)

    # Detect class mapping mismatch: build gt→pred majority vote
    from collections import Counter
    gt_to_preds = defaultdict(list)
    for pred_cls, gt_cls in all_pairs:
        gt_to_preds[gt_cls].append(pred_cls)
    remap = {}
    for gt_cls, preds in sorted(gt_to_preds.items()):
        most_common = Counter(preds).most_common(1)[0][0]
        remap[gt_cls] = most_common
    remapped_correct = sum(1 for pred_cls, gt_cls in all_pairs
                           if pred_cls == remap.get(gt_cls, gt_cls))
    if remapped_correct > cls_correct + 5:
        print(f"\n[WARN] Class ID mismatch detected in test GT labels!")
        print(f"[WARN] The YOLO test labels use different class IDs than GTSRB.")
        print(f"[WARN] Raw accuracy: {cls_correct}/{cls_total} "
              f"-> Remapped accuracy: {remapped_correct}/{cls_total}")
        n_mapped = min(10, len(remap))
        shown = list(sorted(remap.items()))[:n_mapped]
        print(f"[WARN] Sample mapping (GT_label -> GTSRB_class): "
              f"{', '.join(f'{g}->{p}' for g, p in shown)}")
        cls_correct = remapped_correct

    det_p = tp_det / (tp_det + fp_det) if (tp_det + fp_det) > 0 else 0
    det_r = tp_det / (tp_det + fn_det) if (tp_det + fn_det) > 0 else 0
    det_f1 = 2 * det_p * det_r / (det_p + det_r) if (det_p + det_r) > 0 else 0
    cls_acc = cls_correct / cls_total if cls_total > 0 else 0

    print(f"[PIPELINE] Detection: P={det_p:.4f} R={det_r:.4f} F1={det_f1:.4f}")
    print(f"[PIPELINE] Classification of detected signs: {cls_acc:.4f} ({cls_correct}/{cls_total})")
    return {
        "detection": {"precision": round(det_p, 4), "recall": round(det_r, 4),
                      "f1": round(det_f1, 4), "tp": tp_det, "fp": fp_det, "fn": fn_det},
        "classification_of_detected": {"accuracy": round(cls_acc, 4),
                                        "correct": cls_correct, "total": cls_total},
        "test_images": len(test_images),
    }


def main():
    parser = argparse.ArgumentParser(description="Final evaluation on frozen test set")
    parser.add_argument("--config", default="configs/kaggle.yaml")
    parser.add_argument("--classifier-weights", required=True)
    parser.add_argument("--detector-1cls", required=True)
    parser.add_argument("--detector-43cls", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Final evaluation on FROZEN test set. Device: {device}")
    print(f"{'='*60}")

    gt_path = os.path.join(cfg["gtsdb_root"], "gt.txt")
    gt_rows = parse_gtsdb_gt(gt_path) if os.path.exists(gt_path) else []

    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # 1. Classification
    print("\n--- Classification (GTSRB test set) ---")
    report["classification"] = eval_classifier(
        args.classifier_weights, cfg["gtsrb_root"], device
    )

    # 2. Detection 1-class
    print("\n--- Detection 1-class (GTSDB test 600-899) ---")
    report["detection_1cls"] = eval_detector(
        args.detector_1cls, cfg["gtsdb_root"], gt_rows, 1, device
    )

    # 3. Detection 43-class
    if args.detector_43cls:
        print("\n--- Detection 43-class (GTSDB test 600-899) ---")
        report["detection_43cls"] = eval_detector(
            args.detector_43cls, cfg["gtsdb_root"], gt_rows, 43, device
        )

    # 4. A1 Pipeline
    print("\n--- A1 Pipeline: detect-then-classify (GTSDB test 600-899) ---")
    report["pipeline_a1"] = eval_pipeline(
        args.detector_1cls, args.classifier_weights,
        cfg["gtsdb_root"], gt_rows, device
    )

    # Save report
    out_dir = os.path.join(cfg.get("output_dir", "results"), "final_eval")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "final_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[EVAL] Final report saved to {report_path}")


if __name__ == "__main__":
    main()
