"""Compare logit-readout scorers (MC vs README-style Y/N) on ag_news.

usage: uv run python -u -m src.eval [model.gguf] [n_examples]
"""
import sys
import time

import numpy as np
from datasets import load_dataset

from .common import Model
from .gen import gen_baseline
from .mc import score_mc
from .metrics import report
from .yn import score_yn


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    model = Model(path)

    ds = load_dataset("fancyzhx/ag_news", split="test").shuffle(seed=0).select(range(n))
    arts = [t[:600] for t in ds["text"]]
    y = np.array(ds["label"])

    S = {"mc": [], "yn_gap": [], "yn_p": []}
    gen_pred = []
    t_mc = t_yn = t_gen = 0.0
    for i, a in enumerate(arts):
        t = time.perf_counter(); S["mc"].append(score_mc(model, a)); t_mc += time.perf_counter() - t
        t = time.perf_counter(); gap, lp = score_yn(model, a); t_yn += time.perf_counter() - t
        S["yn_gap"].append(gap); S["yn_p"].append(lp)
        t = time.perf_counter(); gen_pred.append(gen_baseline(model, a)); t_gen += time.perf_counter() - t
        if i % 20 == 0:
            print(f"  {i}/{n}", flush=True)

    S = {k: np.array(v) for k, v in S.items()}
    np.savez(f"scores_{path.split('/')[-1]}.npz", y=y, gen=np.array(gen_pred), **S)
    print(f"\nmodel={path} n={n}  (eval on 2nd half; T fitted on 1st half)")
    for k, v in S.items():
        report(k, v, y)
    print(f"gen      acc={np.mean(np.array(gen_pred)[n // 2:] == y[n // 2:]):.3f}")
    print(f"latency/question: mc={t_mc / n * 1000:.0f}ms  yn({len(S['yn_p'][0])} passes)={t_yn / n * 1000:.0f}ms  gen={t_gen / n * 1000:.0f}ms")


if __name__ == "__main__":
    main()
