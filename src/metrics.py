"""Accuracy / NLL / ECE, with a temperature fitted on one half and evaluated on the other."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == y).astype(float)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def nll(s, y, T):
    return -log_softmax(s / T, axis=1)[np.arange(len(y)), y].mean()


def report(name, s, y):
    h = len(y) // 2
    T = minimize_scalar(lambda t: nll(s[:h], y[:h], t), bounds=(0.05, 100), method="bounded").x
    ev, yv = s[h:], y[h:]
    raw, cal = softmax(ev, axis=1), softmax(ev / T, axis=1)
    print(f"{name:8s} acc={np.mean(raw.argmax(1) == yv):.3f}  "
          f"raw: nll={nll(ev, yv, 1):.3f} ece={ece(raw, yv):.3f}   "
          f"T={T:5.2f} cal: nll={nll(ev, yv, T):.3f} ece={ece(cal, yv):.3f}")
