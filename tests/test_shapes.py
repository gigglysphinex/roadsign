from roadsign.shapes import conv_out, conv_params, linear_params, describe, total_params

L = [dict(name="conv1", type="conv", F=32, K=3, S=1, P=1), dict(name="relu1", type="relu"),
     dict(name="conv2", type="conv", F=32, K=3, S=1, P=1), dict(name="relu2", type="relu"),
     dict(name="pool1", type="maxpool", K=2, S=2),
     dict(name="conv3", type="conv", F=64, K=3, S=1, P=1), dict(name="relu3", type="relu"),
     dict(name="pool2", type="maxpool", K=2, S=2),
     dict(name="conv4", type="conv", F=128, K=3, S=1, P=1), dict(name="relu4", type="relu"),
     dict(name="pool3", type="maxpool", K=2, S=2),
     dict(name="gap", type="gap"), dict(name="fc", type="linear", J=43)]


def test_course_formulas():
    assert conv_out(224, 11, 4, 2) == 55          # AlexNet conv1
    assert conv_out(55, 3, 2, 0) == 27            # AlexNet pool1
    assert conv_out(6, 3, 2, 1) == 3              # revision TP conv3 (floor(2.5)+1)
    assert conv_out(4, 2, 1, 0, d=2) == 2         # dilated 2x2 kernel -> K_eff = 3
    assert conv_params(3, 96, 11) == 34944
    assert linear_params(9216, 4096) == 37752832


def test_gtsrb_cnn_table():
    rows = describe(L, (3, 48, 48))
    shapes = {n: s for n, s, _ in rows}
    assert shapes["conv2"] == (32, 48, 48) and shapes["pool1"] == (32, 24, 24)
    assert shapes["pool2"] == (64, 12, 12) and shapes["pool3"] == (128, 6, 6)
    assert shapes["gap"] == (128,) and shapes["fc"] == (43,)
    params = {n: p for n, _, p in rows}
    assert params["conv1"] == 896 and params["conv2"] == 9248 and params["conv3"] == 18496
    assert params["conv4"] == 73856 and params["fc"] == 5547
    assert total_params(rows) == 108043
