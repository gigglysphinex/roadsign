import numpy as np
from roadsign.boxes import iou, iou_matrix, nms, class_aware_nms, xyxy_to_yolo, yolo_to_xyxy


def test_iou_identical_and_disjoint():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_known_values():
    # two 10x10 boxes shifted by 5 in x: inter 50, union 150
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9
    # box inside another: 25 / 100
    assert abs(iou((0, 0, 10, 10), (2, 2, 7, 7)) - 0.25) < 1e-9
    # touching edges: no overlap
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_iou_matrix_shape_and_symmetry():
    a = np.array([[0, 0, 10, 10], [5, 5, 15, 15]]); b = np.array([[0, 0, 10, 10], [6, 6, 12, 12], [50, 50, 60, 60]])
    m = iou_matrix(a, b)
    assert m.shape == (2, 3)
    assert np.allclose(m, iou_matrix(b, a).T)


def test_nms_keeps_best_and_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]])
    scores = np.array([0.8, 0.9, 0.7])
    assert nms(boxes, scores, 0.5) == [1, 2]


def test_nms_threshold_effect():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]])  # IoU = 1/3
    scores = np.array([0.9, 0.8])
    assert nms(boxes, scores, 0.5) == [0, 1]
    assert nms(boxes, scores, 0.3) == [0]


def test_nms_empty_and_single():
    assert nms(np.zeros((0, 4)), np.zeros(0)) == []
    assert nms([[0, 0, 5, 5]], [0.3]) == [0]


def test_class_aware_nms_keeps_different_classes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
    assert class_aware_nms(boxes, np.array([0.9, 0.8]), np.array([1, 2])) == [0, 1]
    assert class_aware_nms(boxes, np.array([0.9, 0.8]), np.array([1, 1])) == [0]


def test_nms_matches_torchvision_if_available():
    try:
        import torch, torchvision
    except ImportError:
        return
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (200, 2)); wh = rng.uniform(5, 40, (200, 2))
    boxes = np.hstack([xy, xy + wh]); scores = rng.uniform(size=200)
    ref = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5).tolist()
    assert nms(boxes, scores, 0.5) == ref


def test_yolo_roundtrip():
    b = (100, 200, 140, 260)
    assert np.allclose(yolo_to_xyxy(xyxy_to_yolo(b, 1360, 800), 1360, 800), b)
