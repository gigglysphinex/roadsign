"""A1 Detect-then-classify pipeline.

Stage 1: YOLOv8 1-class detector finds sign bounding boxes.
Stage 2: RoadSignNet classifier labels each crop (43 classes).
"""
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from .model import RoadSignNet
from .boxes import nms
from .data import GTSDB_SIZE

CLASS_NAMES = {
    0: "speed_20", 1: "speed_30", 2: "speed_50", 3: "speed_60", 4: "speed_70",
    5: "speed_80", 6: "end_speed_80", 7: "speed_100", 8: "speed_120",
    9: "no_overtaking", 10: "no_overtaking_trucks",
    11: "priority_next", 12: "priority_road", 13: "yield", 14: "stop",
    15: "no_vehicles", 16: "no_trucks", 17: "no_entry", 18: "danger",
    19: "curve_left", 20: "curve_right", 21: "double_curve", 22: "bumpy",
    23: "slippery", 24: "narrows_right", 25: "roadwork", 26: "traffic_signals",
    27: "pedestrians", 28: "children", 29: "bicycles", 30: "ice",
    31: "wild_animals", 32: "end_all", 33: "right_only", 34: "left_only",
    35: "straight_only", 36: "straight_right", 37: "straight_left",
    38: "keep_right", 39: "keep_left", 40: "roundabout", 41: "end_no_overtaking",
    42: "end_no_overtaking_trucks",
}

CLASSIFY_TF = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3401, 0.3120, 0.3214],
                         std=[0.2724, 0.2608, 0.2669]),
])


class DetectThenClassify:
    def __init__(self, detector_weights, classifier_weights,
                 device="cpu", det_conf=0.25, nms_iou=0.45, tta=False):
        self.device = torch.device(device)
        self.det_conf = det_conf
        self.nms_iou = nms_iou
        self.tta = tta

        from ultralytics import YOLO
        self.detector = YOLO(detector_weights)

        self.classifier = RoadSignNet(num_classes=43)
        self.classifier.load_state_dict(
            torch.load(classifier_weights, map_location=self.device, weights_only=True)
        )
        self.classifier.to(self.device).eval()

    def predict(self, img):
        """Run full pipeline on a PIL Image or file path.

        Returns list of dicts: {box: (x1,y1,x2,y2), class_id: int,
                                 class_name: str, det_conf: float, cls_conf: float}
        """
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")

        det_results = self.detector(img, conf=self.det_conf, iou=self.nms_iou,
                                     verbose=False, augment=self.tta)
        detections = []
        for r in det_results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu())
                crop = img.crop((max(0, x1), max(0, y1), x2, y2))
                inp = CLASSIFY_TF(crop).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.classifier(inp)
                    probs = torch.softmax(logits, dim=1)
                    cls_conf, cls_id = probs.max(1)
                detections.append({
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                    "class_id": int(cls_id),
                    "class_name": CLASS_NAMES[int(cls_id)],
                    "det_conf": round(conf, 4),
                    "cls_conf": round(float(cls_conf), 4),
                })
        return detections

    def predict_batch(self, image_paths):
        """Run pipeline on a list of image paths. Returns list of (path, detections)."""
        results = []
        for p in image_paths:
            dets = self.predict(p)
            results.append((p, dets))
        return results
