"""Score the multiple-choice readout against greedy generation on a labelled dataset.

usage: scripts/run.sh [model.gguf] [n_examples] [--dataset ag_news|dbpedia_14] [--hide K]
(run.sh runs this in the nix shell, which ROCm needs)

--hide K runs the abstention test instead (see abstain.py): K classes are hidden from the options.
"""
import argparse
import time

import numpy as np

from .common import Model
from .data import DATASETS, load
from .gen import gen_baseline
from .mc import score_mc
from .metrics import report


def print_report(args, classes, mc, gen, y, lat):
    n = len(y)
    print(f"\nmodel={args.model} dataset={args.dataset} ({len(classes)} classes) n={n}  "
          f"(eval on 2nd half; T fitted on 1st half)")
    report("mc", mc, y)
    print(f"gen      acc={np.mean(gen[n // 2:] == y[n // 2:]):.3f}  unparsed={np.mean(gen == -1):.3f}")
    print(f"chance   acc={1 / len(classes):.3f}")
    print("latency/question: " + "  ".join(f"{k}={np.mean(v) * 1000:.0f}ms" for k, v in lat.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="models/Qwen3.5-4B-Q4_K_M.gguf")
    ap.add_argument("n", nargs="?", type=int, default=200)
    ap.add_argument("--dataset", choices=DATASETS, default="ag_news")
    ap.add_argument("--hide", type=int, default=0, metavar="K", help="abstention test with K hidden classes")
    ap.add_argument("--hide-seed", type=int, default=0, help="which K classes get hidden")
    args = ap.parse_args()

    if args.hide:
        from .abstain import run
        return run(args)

    model = Model(args.model)
    arts, y, classes = load(args.dataset, args.n)

    score_mc(model, arts[0], classes); gen_baseline(model, arts[0], classes)  # warm up GPU kernels
    mc, gen, lat = [], [], {"mc": [], "gen": []}
    for i, a in enumerate(arts):
        t = time.perf_counter(); mc.append(score_mc(model, a, classes)); lat["mc"].append(time.perf_counter() - t)
        t = time.perf_counter(); gen.append(gen_baseline(model, a, classes)); lat["gen"].append(time.perf_counter() - t)
        if i % 10 == 0:
            print(f"  {i}/{args.n}", flush=True)

    mc, gen = np.array(mc), np.array(gen)
    name = args.model.split("/")[-1]
    np.savez(f"scores_{args.dataset}_{name}.npz", y=y, gen=gen, mc=mc, **{f"lat_{k}": np.array(v) for k, v in lat.items()})
    print_report(args, classes, mc, gen, y, lat)


if __name__ == "__main__":
    main()
