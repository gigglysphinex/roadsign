"""Output sizes and parameter counts computed with the course formulas (Chapters 2-3).

Used to build the layer table by hand, then to check it against the real PyTorch model
(see check_against_torch).
"""
from math import floor


def conv_out(i, k, s=1, p=0, d=1):
    """O = floor((I + 2P - K_eff) / S) + 1, with K_eff = d(K-1)+1."""
    k_eff = d * (k - 1) + 1
    return floor((i + 2 * p - k_eff) / s) + 1


def conv_params(c_in, f, k, bias=True):
    return f * (c_in * k * k + (1 if bias else 0))


def linear_params(i, j, bias=True):
    return j * (i + (1 if bias else 0))


def bn_params(c):
    """Trainable parameters of BatchNorm2d: gamma and beta."""
    return 2 * c


def describe(layers, in_shape):
    """layers: list of dicts {name, type, ...}. Returns table rows (name, out_shape, params).

    type in: conv(F,K,S,P) | bn | relu | maxpool(K,S,P) | gap | flatten | dropout | linear(J)
    """
    c, h, w = in_shape
    flat = None
    rows = []
    for L in layers:
        t, params = L["type"], 0
        if t == "conv":
            params = conv_params(c, L["F"], L["K"])
            h, w, c = conv_out(h, L["K"], L.get("S", 1), L.get("P", 0)), conv_out(w, L["K"], L.get("S", 1), L.get("P", 0)), L["F"]
        elif t == "bn":
            params = bn_params(c)
        elif t == "maxpool":
            h, w = conv_out(h, L["K"], L["S"], L.get("P", 0)), conv_out(w, L["K"], L["S"], L.get("P", 0))
        elif t == "gap":
            h = w = 1
            flat = c
        elif t == "flatten":
            flat = c * h * w
        elif t == "linear":
            params = linear_params(flat, L["J"])
            flat = L["J"]
        out = (flat,) if flat is not None else (c, h, w)
        rows.append((L["name"], out, params))
    return rows


def total_params(rows):
    return sum(r[2] for r in rows)


def check_against_torch(model, rows):
    """Compare the hand-computed total with sum(p.numel()) of a torch model."""
    real = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return real, total_params(rows), real == total_params(rows)
