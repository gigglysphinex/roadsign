import cv2
import numpy as np
import pytest
from roadsign.data import (parse_gtsdb_gt, anomaly_report, block_split, gtsrb_track_split,
                           to_yolo_lines, class_counts, assert_not_test, is_gtsdb_test_image)
from roadsign.synth import paste_signs
from roadsign.baseline import color_shape_candidates, detection_recall, train_hog_svm, predict_hog_svm
from roadsign.boxes import iou_matrix


def test_parse_and_anomalies(tmp_path):
    p = tmp_path / "gt.txt"
    p.write_text("00000.ppm;774;411;815;446;11\n00000.ppm;774;411;815;446;11\n"
                 "00001.ppm;1300;100;1400;150;3\n00002.ppm;10;10;5;50;2\n00003.ppm;1;1;30;30;60\n")
    rows = parse_gtsdb_gt(p)
    assert len(rows) == 5 and rows[0]["cls"] == 11 and rows[0]["box"] == (774, 411, 815, 446)
    counts, rep = anomaly_report(rows, all_images=["00000.ppm", "00001.ppm", "00002.ppm", "00003.ppm", "00004.ppm"])
    assert counts["duplicate"] == 1 and counts["box_outside_image"] == 1
    assert counts["degenerate_box"] == 1 and counts["class_out_of_range"] == 1
    assert rep["image_without_annotation"] == ["00004.ppm"]
    assert class_counts(rows)[11] == 2


def test_block_split_no_overlap_and_block_integrity():
    names = [f"{i:05d}.ppm" for i in range(600)]
    tr, va = block_split(names, 0.2, block=20, seed=1)
    assert set(tr).isdisjoint(va) and len(tr) + len(va) == 600
    # every block of 20 consecutive images lies entirely in one side
    side = {n: "v" if n in set(va) else "t" for n in names}
    for s in range(0, 600, 20):
        assert len({side[n] for n in names[s:s + 20]}) == 1
    assert block_split(names, 0.2, 20, 1) == (tr, va)  # deterministic


def test_gtsrb_track_split_keeps_tracks_together():
    files = [f"{c:05d}/{t:05d}_{f:05d}.ppm" for c in range(3) for t in range(10) for f in range(30)]
    tr, va = gtsrb_track_split(files, 0.2, seed=0)
    tracks = lambda fs: {(f.split("/")[0], f.split("/")[1].split("_")[0]) for f in fs}
    assert tracks(tr).isdisjoint(tracks(va)) and len(tr) + len(va) == len(files)
    assert all(any(f.startswith(f"{c:05d}") for f in va) for c in range(3))  # each class in val


def test_yolo_lines():
    rows = [dict(img="a", box=(0, 0, 136, 80), cls=7)]
    assert to_yolo_lines(rows, 1360, 800) == ["7 0.050000 0.050000 0.100000 0.100000"]
    assert to_yolo_lines(rows, 1360, 800, single_class=True)[0].startswith("0 ")


def test_paste_signs_no_overlap_and_boxes_in_image():
    rng = np.random.default_rng(0)
    bg = rng.integers(0, 255, (400, 600, 3), dtype=np.uint8)
    crops = [np.full((40, 40, 3), 200, np.uint8) for _ in range(5)]
    real = [(100, 100, 150, 150)]
    img, boxes, labels = paste_signs(bg, crops, [0, 1, 2, 3, 4], real, rng, n_signs=5)
    assert img.shape == bg.shape and len(boxes) == len(labels) > 0
    allb = np.array(real + boxes)
    m = iou_matrix(allb, allb); np.fill_diagonal(m, 0)
    assert (m == 0).all()
    assert all(0 <= b[0] < b[2] <= 600 and 0 <= b[1] < b[3] <= 400 for b in boxes)


def test_color_candidates_finds_red_disc():
    img = np.full((200, 300, 3), 128, np.uint8)
    cv2.circle(img, (150, 100), 30, (0, 0, 220), -1)  # red in BGR
    boxes = color_shape_candidates(img)
    rec, prec = detection_recall(boxes, [(120, 70, 180, 130)])
    assert rec == 1.0 and prec == 1.0


def test_assert_not_test_blocks_gtsdb_test_images():
    # These should raise
    for name in ["00600.ppm", "00750.ppm", "00899.ppm", "data/gtsdb/00600.ppm"]:
        with pytest.raises(ValueError, match="FORBIDDEN"):
            assert_not_test(name)
    # These should be fine (training range)
    for name in ["00000.ppm", "00599.ppm", "data/gtsdb/00100.ppm"]:
        assert_not_test(name)  # no error


def test_assert_not_test_blocks_gtsrb_test_folder():
    with pytest.raises(ValueError, match="FORBIDDEN"):
        assert_not_test("data/gtsrb/Final_Test/00001/00000_00000.ppm")
    # Training folder is fine
    assert_not_test("data/gtsrb/Train/00001/00000_00000.ppm")


def test_is_gtsdb_test_image():
    assert is_gtsdb_test_image("00600.ppm") is True
    assert is_gtsdb_test_image("00899.ppm") is True
    assert is_gtsdb_test_image("00599.ppm") is False
    assert is_gtsdb_test_image("00900.ppm") is False
    assert is_gtsdb_test_image("not_a_number.ppm") is False


def test_hog_svm_separates_two_shapes():
    rng = np.random.default_rng(0)
    def disc(): 
        im = np.zeros((32, 32, 3), np.uint8); cv2.circle(im, (16, 16), int(rng.integers(8, 12)), (255, 255, 255), -1); return im
    def sq():
        im = np.zeros((32, 32, 3), np.uint8); cv2.rectangle(im, (6, 6), (int(rng.integers(22, 27)),) * 2, (255, 255, 255), -1); return im
    X = [disc() for _ in range(30)] + [sq() for _ in range(30)]; y = [0] * 30 + [1] * 30
    model = train_hog_svm(X, y)
    assert (predict_hog_svm(model, [disc(), sq()]) == [0, 1]).all()
