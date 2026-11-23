#!/usr/bin/env python3
"""EF5 Demonstrator: feed an image, get annotated output with detections.

Draws bounding boxes and class labels on the input image using the
A1 detect-then-classify pipeline.

Usage:
    python scripts/demo.py \
        --image data/gtsdb/00042.jpg \
        --detector results/detection/yolov8n_1cls/train/weights/best.pt \
        --classifier results/classification/scratch_aug1_lr0.001/best.pt \
        --output results/demo/00042_annotated.jpg

    # Batch mode: annotate all images in a directory
    python scripts/demo.py \
        --image-dir data/gtsdb/ \
        --detector ... --classifier ... --output-dir results/demo/
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def draw_detections(img, detections):
    """Draw boxes and labels on a PIL Image. Returns annotated copy."""
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    colors = {
        "prohibitory": "#E74C3C",
        "mandatory": "#3498DB",
        "danger": "#F39C12",
        "other": "#2ECC71",
    }

    def sign_category(cls_id):
        if cls_id <= 10 or cls_id in (15, 16, 17):
            return "prohibitory"
        if cls_id >= 33 and cls_id <= 40:
            return "mandatory"
        if cls_id >= 18 and cls_id <= 31:
            return "danger"
        return "other"

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cat = sign_category(det["class_id"])
        color = colors[cat]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{det['class_name']} {det['cls_conf']:.0%}"
        bbox = font.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(0, y1 - th - 6)
        draw.rectangle([x1, label_y, x1 + tw + 8, label_y + th + 6], fill=color)
        draw.text((x1 + 4, label_y + 2), label, fill="white", font=font)

    return img


def main():
    parser = argparse.ArgumentParser(description="RoadSign demonstrator")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--image-dir", help="Directory of images for batch mode")
    parser.add_argument("--detector", required=True, help="YOLOv8 1-class weights")
    parser.add_argument("--classifier", required=True, help="RoadSignNet weights")
    parser.add_argument("--output", help="Output path for single image")
    parser.add_argument("--output-dir", default="results/demo",
                        help="Output directory for batch mode")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        print("ERROR: Provide --image or --image-dir")
        sys.exit(1)

    import torch
    from PIL import Image
    from roadsign.pipeline import DetectThenClassify

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DetectThenClassify(
        args.detector, args.classifier,
        device=device, det_conf=args.conf
    )

    if args.image:
        images = [args.image]
    else:
        exts = (".jpg", ".jpeg", ".png", ".ppm")
        images = sorted([
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.lower().endswith(exts)
        ])

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        dets = pipe.predict(img)

        annotated = draw_detections(img.copy(), dets)

        if args.image and args.output:
            out_path = args.output
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        else:
            stem = Path(img_path).stem
            out_path = os.path.join(args.output_dir, f"{stem}_annotated.jpg")

        annotated.save(out_path, quality=95)

        print(f"[DEMO] {os.path.basename(img_path)}: {len(dets)} signs detected -> {out_path}")
        for d in dets:
            print(f"       {d['class_name']} ({d['cls_conf']:.0%}) at {d['box']}")

        all_results.append({
            "image": os.path.basename(img_path),
            "detections": dets,
        })

    json_path = os.path.join(args.output_dir, "demo_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[DEMO] Results saved to {json_path}")
    print(f"[DEMO] {len(images)} images processed, "
          f"{sum(len(r['detections']) for r in all_results)} total signs detected.")


if __name__ == "__main__":
    main()
