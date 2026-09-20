"""Compare logit-readout scorers (MC vs README-style Y/N) against greedy generation.

usage: scripts/run.sh [model.gguf] [n_examples] [--dataset ag_news|dbpedia_14] [--no-yn]
(run.sh runs this in the nix shell, which ROCm needs)
"""
import argparse
import time

import numpy as np

from .common import Model
from .data import DATASETS, load
from .gen import gen_baseline
from .mc import score_mc
from .metrics import report
from .yn import VARIANTS, score_yn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="models/Qwen3.5-4B-Q4_K_M.gguf")
    ap.add_argument("n", nargs="?", type=int, default=200)
    ap.add_argument("--dataset", choices=DATASETS, default="ag_news")
    ap.add_argument("--no-yn", action="store_true", help="skip the per-option Y/N scorer (one pass per class)")
    args = ap.parse_args()

    model = Model(args.model)
    arts, y, classes = load(args.dataset, args.n)
    run_yn = not args.no_yn

    # warm up GPU kernels so compilation does not count as latency
    score_mc(model, arts[0], classes); gen_baseline(model, arts[0], classes)
    if run_yn:
        score_yn(model, arts[0], classes)

    S = {"mc": []}
    if run_yn:
        S.update({f"yn_{v}": [] for v in VARIANTS})
    mass, gen_pred = [], []
    lat = {"mc": [], "gen": [], **({"yn": []} if run_yn else {})}

    def timed(key, fn, *a):
        t = time.perf_counter(); r = fn(*a); lat[key].append(time.perf_counter() - t)
        return r

    for i, a in enumerate(arts):
        S["mc"].append(timed("mc", score_mc, model, a, classes))
        if run_yn:
            variants, m = timed("yn", score_yn, model, a, classes)
            for v in VARIANTS:
                S[f"yn_{v}"].append(variants[v])
            mass.append(m)
        gen_pred.append(timed("gen", gen_baseline, model, a, classes))
        if i % 10 == 0:
            print(f"  {i}/{args.n}", flush=True)

    S = {k: np.array(v) for k, v in S.items()}
    name = args.model.split("/")[-1]
    np.savez(f"scores_{args.dataset}_{name}.npz", y=y, gen=np.array(gen_pred),
             **({"mass": np.array(mass)} if run_yn else {}),
             **{f"lat_{k}": np.array(v) for k, v in lat.items()}, **S)

    n = args.n
    print(f"\nmodel={args.model} dataset={args.dataset} ({len(classes)} classes) n={n}  "
          f"(eval on 2nd half; T fitted on 1st half)")
    for k, v in S.items():
        report(k, v, y)
    print(f"gen      acc={np.mean(np.array(gen_pred)[n // 2:] == y[n // 2:]):.3f}  unparsed={np.mean(np.array(gen_pred) == -1):.3f}")
    print(f"chance   acc={1 / len(classes):.3f}")
    if run_yn:
        print(f"mass on {{Y,N}}: mean={np.mean(mass):.3f} min={np.min(mass):.3f}")
    print("latency/question: " + "  ".join(f"{k}={np.mean(v) * 1000:.0f}ms" for k, v in lat.items()))


if __name__ == "__main__":
    main()
