"""Abstention test: does an explicit "None of the above" option beat rejecting low-confidence answers?

Some classes are hidden from the options; their examples are out of scope and the right answer
for them is to abstain. Two arms are scored on the same examples:
  A: the visible classes plus a "None of the above" option (abstain = predict that option)
  B: the visible classes only (abstain = top probability below a threshold)
"""
import time

import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

from .common import Model
from .data import hide_classes, load
from .mc import score_mc

FALSE_ABSTAIN = 0.05  # operating point for the fair comparison: abstain on 5% of in-scope examples


def recall_at_false_abstain(score, oos, rate=FALSE_ABSTAIN):
    """Out-of-scope recall when the threshold abstains on `rate` of the in-scope examples.
    score: higher = more likely out of scope."""
    thr = np.quantile(score[~oos], 1 - rate)
    return float(np.mean(score[oos] > thr))


def report(A, B, y, y_full, classes, hidden):
    """A: scores with the abstain option last; B: scores over the visible classes only.
    y: labels over the visible classes, -1 = out of scope; y_full: original labels; hidden: their ids."""
    oos = y < 0
    none = A.shape[1] - 1
    pa, pb = softmax(A, axis=1), softmax(B, axis=1)
    a_pred = pa.argmax(1)
    s_a, s_b = pa[:, none], 1 - pb.max(1)  # out-of-scope scores: p(None) vs low top probability

    print(f"in scope: {(~oos).sum()}  out of scope: {oos.sum()}  hidden: {', '.join(classes[h] for h in hidden)}")
    print(f"in-scope accuracy   B (no option): {np.mean(pb[~oos].argmax(1) == y[~oos]):.3f}   "
          f"A (with option): {np.mean(a_pred[~oos] == y[~oos]):.3f}")
    print(f"arm A as it stands: abstention recall={np.mean(a_pred[oos] == none):.3f}  "
          f"false-abstain rate={np.mean(a_pred[~oos] == none):.3f}")
    print(f"separating in from out of scope:  AUROC  A p(None)={roc_auc_score(oos, s_a):.3f}  "
          f"B 1-top-prob={roc_auc_score(oos, s_b):.3f}")
    print(f"recall at {FALSE_ABSTAIN:.0%} false abstains:  A={recall_at_false_abstain(s_a, oos):.3f}  "
          f"B={recall_at_false_abstain(s_b, oos):.3f}")
    print("abstention recall per hidden class (arm A):")
    for h in hidden:
        m = y_full == h
        print(f"  {classes[h]:26s} n={m.sum():3d}  recall={np.mean(a_pred[m] == none):.3f}")


def run(args):
    model = Model(args.model)
    arts, y_full, classes = load(args.dataset, args.n)
    visible, y, hidden = hide_classes(classes, y_full, args.hide, args.hide_seed)

    score_mc(model, arts[0], visible, abstain=True); score_mc(model, arts[0], visible)  # warm up
    A, B, lat = [], [], []
    for i, a in enumerate(arts):
        t = time.perf_counter()
        A.append(score_mc(model, a, visible, abstain=True))
        B.append(score_mc(model, a, visible))
        lat.append(time.perf_counter() - t)
        if i % 10 == 0:
            print(f"  {i}/{args.n}", flush=True)
    A, B = np.array(A), np.array(B)

    name = args.model.split("/")[-1]
    np.savez(f"scores_{args.dataset}_hide{args.hide}s{args.hide_seed}_{name}.npz",
             A=A, B=B, y=y, y_full=y_full, hidden=np.array(hidden), lat=np.array(lat))
    print(f"\nmodel={args.model} dataset={args.dataset} n={args.n} hidden={args.hide} (seed {args.hide_seed})")
    report(A, B, y, y_full, classes, hidden)
    print(f"latency/question (both arms): {np.mean(lat) * 1000:.0f}ms")
