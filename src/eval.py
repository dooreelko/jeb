"""Compare logit-readout scorers (MC vs README-style Y/N) on ag_news.

usage: scripts/run.sh [model.gguf] [n_examples]      (runs this in the nix shell, needed for ROCm)
"""
import sys
import time

import numpy as np
from datasets import load_dataset

from .common import Model
from .gen import gen_baseline
from .mc import score_mc
from .metrics import report
from .yn import VARIANTS, score_yn


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3.5-4B-Q4_K_M.gguf"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    model = Model(path)

    ds = load_dataset("fancyzhx/ag_news", split="test").shuffle(seed=0).select(range(n))
    arts = [t[:600] for t in ds["text"]]
    y = np.array(ds["label"])

    score_mc(model, arts[0]); score_yn(model, arts[0]); gen_baseline(model, arts[0])  # warm up GPU kernels

    S = {"mc": [], **{f"yn_{v}": [] for v in VARIANTS}}
    mass, gen_pred = [], []
    lat = {"mc": [], "yn": [], "gen": []}

    def timed(key, fn, *a):
        t = time.perf_counter(); r = fn(*a); lat[key].append(time.perf_counter() - t)
        return r

    for i, a in enumerate(arts):
        S["mc"].append(timed("mc", score_mc, model, a))
        variants, m = timed("yn", score_yn, model, a)
        for v in VARIANTS:
            S[f"yn_{v}"].append(variants[v])
        mass.append(m)
        gen_pred.append(timed("gen", gen_baseline, model, a))
        if i % 10 == 0:
            print(f"  {i}/{n}", flush=True)

    S = {k: np.array(v) for k, v in S.items()}
    name = path.split("/")[-1]
    np.savez(f"scores_{name}.npz", y=y, gen=np.array(gen_pred), mass=np.array(mass),
             **{f"lat_{k}": np.array(v) for k, v in lat.items()}, **S)

    print(f"\nmodel={path} n={n}  (eval on 2nd half; T fitted on 1st half)")
    for k, v in S.items():
        report(k, v, y)
    print(f"gen      acc={np.mean(np.array(gen_pred)[n // 2:] == y[n // 2:]):.3f}  unparsed={np.mean(np.array(gen_pred) == -1):.3f}")
    print(f"mass on {{Y,N}}: mean={np.mean(mass):.3f} min={np.min(mass):.3f}")
    print("latency/question: " + "  ".join(f"{k}={np.mean(v) * 1000:.0f}ms" for k, v in lat.items()))


if __name__ == "__main__":
    main()
