# RoadSign – Advanced Computer Vision mini-project (subject C)

Detect and recognise traffic signs: GTSDB for detection, GTSRB for classification.
Enhancements: (1) detect-then-classify pipeline, (2) copy-paste synthetic data, (3) ONNX / INT8 embedded benchmark.

## Layout
```
src/roadsign/boxes.py     IoU, NMS, class-aware NMS, YOLO <-> xyxy        (Chapter 4, written by us)
src/roadsign/shapes.py    output sizes + parameter counts, course formulas (Chapters 2-3)
src/roadsign/data.py      gt parsing, anomaly report, leak-free splits, YOLO labels (EF1)
src/roadsign/synth.py     copy-paste synthetic training images           (enhancement 2)
src/roadsign/baseline.py  HOG+SVM and colour/shape detector, no DL       (Chapter 1)
tests/                    18 unit tests: `python -m pytest -q`
docs/cadrage.md           week-1 framing note (2 pages)
```

## Rules built into the code
- The GTSDB test images (00600-00899) are never read by any of these modules.
- Validation comes from the training data only. GTSDB: `block_split` (blocks of consecutive images).
  GTSRB: `gtsrb_track_split` (all 30 frames of one physical sign stay on the same side).
- Synthetic images use training backgrounds and training crops only, and are checked on the real validation split.
- Splits are saved as JSON (`save_split`) and reloaded, never recomputed.

## Run order (Colab / Kaggle GPU)
1. `pip install -r requirements.txt`, then `python -m pytest -q` (all green before anything else).
2. Download GTSDB and GTSRB, run the annotation report and save the splits (EF1).
3. Baseline on validation (HOG+SVM, colour/shape detector).
4. Classification: CNN from scratch (table in `shapes.describe`, checked with `check_against_torch`) vs pre-trained, one factor at a time.
5. Detection: YOLO (single stage) vs Faster R-CNN (two stage) on the same split; then the detect-then-classify variant.
6. Post-processing + JSON output, demo, ONNX/INT8 benchmark, then the test set once.
