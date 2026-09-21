"""Turn-based Flappy Bird as a decision benchmark for Jev.

The game state is described in words (the context) and the model picks one of two options; the
game waits for the decision, so latency does not affect the score. Physics follow openjev's
Flappy Bird so scores are roughly comparable (900 steps = up to 28 pipes).

usage: scripts/flappy.sh [--policy jev|oracle|random|never] [--options actions|position]
                         [--episodes N] [--max-steps M] [--watch] [--model path.gguf]
       scripts/flappy.sh --agree N   how often the model agrees with the oracle on N visited states
"""
import argparse
import random
import sys
import time

import numpy as np

GRAVITY, FLAP_V, SPEED = 0.003, 0.015, 0.015
PIPE_EVERY, GAP, PIPE_HW, BIRD_X, BIRD_HW = 0.45, 0.28, 0.04, 0.2, 0.03


class Flappy:
    def __init__(self, seed=0, max_steps=900):
        self.rng, self.max_steps = random.Random(seed), max_steps
        self.y, self.vy, self.t, self.score, self.done = 0.5, 0.0, 0, 0, False
        self.pipes = [[1.2, self._gap()]]  # [x, gap_lo]

    def _gap(self):
        return self.rng.uniform(0.15, 0.85 - GAP)

    def next_pipe(self):
        return next((x, lo) for x, lo in self.pipes if x + PIPE_HW >= BIRD_X - BIRD_HW)

    def step(self, flap):
        self.vy = FLAP_V if flap else self.vy - GRAVITY
        self.y += self.vy
        self.t += 1
        for p in self.pipes:
            p[0] -= SPEED
        if self.pipes[-1][0] < 1.2 - PIPE_EVERY:
            self.pipes.append([self.pipes[-1][0] + PIPE_EVERY, self._gap()])
        if self.pipes[0][0] + PIPE_HW < BIRD_X - BIRD_HW:
            self.pipes.pop(0)
            self.score += 1
        x, lo = self.next_pipe()
        hit = self.y <= 0 or self.y >= 1 or (abs(x - BIRD_X) < PIPE_HW + BIRD_HW and not lo < self.y < lo + GAP)
        self.done = hit or self.t >= self.max_steps

    def state(self):
        x, lo = self.next_pipe()
        return {"y": self.y, "vy": self.vy, "dx": x - BIRD_X, "lo": lo, "hi": lo + GAP}


# ----------------------------------------------------------------------------- what the model sees
def centre(s):
    return (s["lo"] + s["hi"]) / 2


TRACE = [10]
DROP = set()  # parts of the state text to leave out, for the prompt ablation: pos, goal, offset


def numbers(s):
    """openjev's `base` state text, verbatim (code/flappy.py render_text), minus anything in DROP."""
    pos = "inside the gap" if s["lo"] < s["y"] < s["hi"] else ("above the gap" if s["y"] >= s["hi"] else "below the gap")
    move = "rising" if s["vy"] > 0 else "falling"
    text = (f"Flappy Bird. The bird is at height {s['y']:.2f} (0 = ground, 1 = ceiling) and is {move} "
            f"with vertical velocity {s['vy']:+.3f} per frame; gravity pulls it down every frame and flapping pushes it up. "
            f"The next pipe is {s['dx']:.2f} ahead; its gap spans heights {s['lo']:.2f} to {s['hi']:.2f} (centre {centre(s):.2f}). ")
    rel = f"{s['y'] - centre(s):+.2f} relative to the gap centre"
    if "pos" in DROP and "offset" in DROP:
        pass
    elif "pos" in DROP:
        text += f"The bird is {rel}. "
    elif "offset" in DROP:
        text += f"The bird is currently {pos}. "
    else:
        text += f"The bird is currently {pos}, {rel}. "
    if "goal" not in DROP:
        text += "The bird must fly through the gap without touching the pipe, the ground or the ceiling."
    return text.strip()


def words(s):
    """The same state with the comparisons already made, and no numbers. A hint, not a fair test of
    reading numbers: the answer to the position question is stated outright."""
    edge = " The bird is close to the ground." if s["y"] < 0.15 else " The bird is close to the ceiling." if s["y"] > 0.85 else ""
    return (f"Flappy Bird. Gravity pulls the bird down every frame and flapping pushes it up. "
            f"The centre of the gap is {'above' if centre(s) > s['y'] else 'below'} the bird. "
            f"The bird is {'rising' if s['vy'] > 0 else 'falling'}. "
            f"The next pipe is {'close' if s['dx'] < 0.2 else 'far'}.{edge}")


def offset_bucket(s):
    """Where the bird is against the gap centre, as a name (computed in code, not left to the model)."""
    d = s["y"] - centre(s)
    for edge, name in ((-0.15, "far below"), (-0.05, "below"), (0, "slightly below"), (0.05, "slightly above"), (0.15, "above")):
        if d < edge:
            return name
    return "far above"


def motion_bucket(s):
    v = s["vy"]
    return ("rising fast" if v > 0.006 else "rising slowly" if v > 0.0015 else "hovering" if v > -0.0015
            else "falling slowly" if v > -0.006 else "falling fast")


SUFFIX = [""]  # context added after the semantic state, set by --style
# what the options say, what they are called, and what the context adds
STYLES = {
    "position": (["The bird is below the centre of the gap", "The bird is above the centre of the gap"], "statement", ""),
    "actions": (["Flap", "Do nothing"], "action", " Flapping pushes the bird up; otherwise it falls."),
    "goal": (["Flap", "Do nothing"], "action", " Flapping pushes the bird up; otherwise it falls. "
             "The bird wants to stay level with the centre of the gap."),
    "rule": (["Flap", "Do nothing"], "action", " Flapping pushes the bird up; otherwise it falls. "
             "Rule: flap when the bird is below the centre of the gap, otherwise do nothing."),
}


def semantic(s):
    """Only the fields the decision needs, as named buckets and no numbers."""
    return f"Flappy Bird. The bird is {offset_bucket(s)} the centre of the gap and is {motion_bucket(s)}.{SUFFIX[0]}"


def describe(s, mode="numbers"):
    """The game state in words. numbers: heights and velocity; words: hints only (ablation)."""
    if mode == "numbers":
        return numbers(s)
    if mode == "words":
        return words(s)
    if mode == "semantic":
        return semantic(s)
    raise ValueError(mode)


def below_centre(s):
    """What "The bird is below the centre of the gap" says; true means flap. The target for reading the state."""
    return s["y"] < centre(s)


# option 0 always means flap. "position" states a fact about the game, "actions" names the move.
OPTIONS = {"actions": ["flap", "do nothing"],
           "position": ["The bird is below the centre of the gap.", "The bird is above the centre of the gap."]}  # openjev's `position` hypotheses


def oracle(s, margin=0.03, lookahead=3):
    """Hand-written reference: flap if the height a few frames ahead falls below the gap centre."""
    y_pred = s["y"] + s["vy"] * lookahead - GRAVITY * lookahead * (lookahead - 1) / 2
    return y_pred < (s["lo"] + s["hi"]) / 2 - margin


# ----------------------------------------------------------------------------- terminal view
def draw(env, action=None, probs=None, w=60, h=18):
    grid = [[" "] * w for _ in range(h)]
    for x, lo in env.pipes:
        col = int(x / 1.0 * (w - 1))
        for c in range(max(col - 1, 0), min(col + 2, w)):
            for r in range(h):
                if not lo < 1 - r / (h - 1) < lo + GAP:
                    grid[r][c] = "#"
    grid[min(max(int(round((1 - env.y) * (h - 1))), 0), h - 1)][int(BIRD_X * (w - 1))] = ">"
    tag = "" if action is None else f"   action: {'FLAP' if action else '----'}"
    if probs is not None:
        tag += f"   p(flap)={probs[0]:.2f}"
    return "\n".join([f"score {env.score:2d}  step {env.t:4d}{tag}", "=" * w, *("".join(r) for r in grid), "=" * w])


# ----------------------------------------------------------------------------- policies
def make_policy(name, args, rng):
    if name == "random":
        return lambda s: ((f := rng.random() < 0.1), [float(f), 1.0 - f])
    if name == "never":
        return lambda s: (False, [0.0, 1.0])
    if name == "oracle":
        return lambda s: ((f := oracle(s)), [float(f), 1.0 - f])
    if name == "rule":  # a perfect reader of the position statement
        return lambda s: ((f := below_centre(s)), [float(f), 1.0 - f])
    from .jev import Jev
    jev = Jev(args.model)
    opts = OPTIONS[args.options]
    if args.style:
        opts, args.noun, SUFFIX[0] = STYLES[args.style]
    if args.no_period:
        opts = [o.rstrip(".") for o in opts]
    thr = [args.flap_at]

    def policy(s):
        p = jev.probabilities(describe(s, args.state), opts, noun=args.noun)
        if args.sample:  # sample the move from the softmax at this temperature instead of taking the argmax
            d = np.log(max(p[0], 1e-9) / max(p[1], 1e-9)) / args.sample
            return bool(rng.random() < 1 / (1 + np.exp(-d))), p
        return bool(p[0] >= thr[0]), p
    if args.threshold:  # fitted on states from other episodes than the ones played, then frozen
        pf, ps = agreement(policy, 100, base=50000)
        cands = np.unique(np.r_[pf, ps])
        thr[0] = float(max(cands, key=lambda c: np.mean(pf >= c) + np.mean(ps < c)))
        print(f"threshold fitted on {len(pf) + len(ps)} separate states: p(flap) >= {thr[0]:.3f}", flush=True)
    return policy


def play(seed, policy, max_steps, watch=False, delay=0.05):
    env, lat = Flappy(seed, max_steps), []
    on_flap = []  # actions taken while the bird is below the gap centre
    tail = []  # (tick, height minus centre, vy, p(flap), flapped) for the trace before death
    ticks = []  # (bird height minus gap centre, flapped) for the error analysis
    while not env.done:
        s = env.state()
        t = time.perf_counter()
        flap, probs = policy(s)
        lat.append(time.perf_counter() - t)
        ticks.append((s["y"] - centre(s), bool(flap)))
        tail.append((env.t, s["y"] - centre(s), s["vy"], float(probs[0]), bool(flap)))
        if below_centre(s):
            on_flap.append(flap)
        if watch:
            print("\033[H\033[J" + draw(env, flap, probs), flush=True)
            time.sleep(delay)
        env.step(flap)
    if watch:
        print("\033[H\033[J" + draw(env), f"\n\nGAME OVER  score {env.score}", flush=True)
    return {"score": env.score, "steps": env.t, "lat_ms": 1000 * float(np.mean(lat)),
            "flap_recall": float(np.mean(on_flap)) if on_flap else float("nan"), "n_flap": len(on_flap),
            "ticks": ticks, "tail": tail, "death": (env.y, env.state()["dx"], ticks[-1][0])}


def agreement(policy, n, seed=0, noise=0.15, base=1000):
    """Agreement with the statement "the bird is below the centre of the gap" on n states, half where it is
    true and half where it is false. (Plain agreement over visited states rewards never flapping.)
    The states come from a noisy oracle playing, so they cover recoveries as well as steady flight.
    Returns p(flap) on the flap states and on the no-flap states."""
    rng, flap_states, stay_states, ep = random.Random(seed), [], [], 0
    while min(len(flap_states), len(stay_states)) < n // 2:
        env = Flappy(base + ep, 600)
        ep += 1
        while not env.done:
            s = env.state()
            a_or = oracle(s)
            (flap_states if below_centre(s) else stay_states).append(s)
            env.step(not a_or if rng.random() < noise else a_or)
    half = n // 2
    pf = np.array([policy(s)[1][0] for s in flap_states[:half]])  # p(flap) where the bird is below the centre
    ps = np.array([policy(s)[1][0] for s in stay_states[:half]])  # p(flap) where it is above
    return pf, ps


def report_agreement(name, pf, ps):
    from sklearn.metrics import roc_auc_score
    print(f"{name}: {len(pf) + len(ps)} balanced states")
    print(f"  at threshold 0.5: agrees when the bird is below the centre {np.mean(pf > .5):.3f}, when above {np.mean(ps <= .5):.3f}, "
          f"balanced accuracy {(np.mean(pf > .5) + np.mean(ps <= .5)) / 2:.3f}  (chance 0.5)")
    print(f"  mean p(flap): below {pf.mean():.2f}, above {ps.mean():.2f}")
    print(f"  AUROC of p(flap) separating the two: {roc_auc_score([1] * len(pf) + [0] * len(ps), np.r_[pf, ps]):.3f}")
    # threshold fitted on every other state, scored on the rest, so it is not tuned on its own test set
    tr = np.arange(len(pf)) % 2 == 0
    cands = np.unique(np.r_[pf[tr], ps[tr]])
    best = max(cands, key=lambda t: (np.mean(pf[tr] >= t) + np.mean(ps[tr] < t)) / 2)
    print(f"  threshold fitted on half ({best:.3f}), scored on the other half: "
          f"balanced accuracy {(np.mean(pf[~tr] >= best) + np.mean(ps[~tr] < best)) / 2:.3f}")


def report_errors(res, edges=(-1, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 1)):
    """Wrong decisions (flap while above the centre, or not while below) by distance to the gap centre."""
    d = np.array([t[0] for r in res for t in r["ticks"]])
    wrong = np.array([t[1] != (t[0] < 0) for r in res for t in r["ticks"]])
    print("wrong decisions by bird height minus gap centre (negative = below):")
    for lo, hi in zip(edges, edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.any():
            print(f"  [{lo:+.2f}, {hi:+.2f})  n={m.sum():4d}  wrong {wrong[m].mean():.2f}")
    print(f"  overall wrong {wrong.mean():.3f} over {len(d)} decisions")
    for i, r in enumerate(res):
        if r["steps"] < 900:
            print(f"episode {i} last decisions (tick, height-centre, vy, p(flap), flapped):")
            for tk, dv, vy, pf, fl in r["tail"][-TRACE[0]:]:
                print(f"  {tk:4d}  {dv:+.3f}  {vy:+.3f}  {pf:.2f}  {'FLAP' if fl else '-'}")
    print("deaths (height, distance to pipe, height minus centre): "
          + "; ".join(f"({r['death'][0]:.2f}, {r['death'][1]:+.2f}, {r['death'][2]:+.2f})" for r in res if r["steps"] < 900))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["jev", "oracle", "rule", "random", "never"], default="jev")
    ap.add_argument("--options", choices=list(OPTIONS), default="position")
    ap.add_argument("--state", choices=["numbers", "words", "semantic"], default="numbers",
                    help="how the state is described; words replaces the numbers with hints derived from the true state (ablation)")
    ap.add_argument("--drop", default="", help="prompt ablation: comma list of pos, goal, offset left out of the state text")
    ap.add_argument("--no-period", action="store_true", help="prompt ablation: options without the final period")
    ap.add_argument("--threshold", action="store_true", help="fit the flap threshold on separate states (calibrated row)")
    ap.add_argument("--sample", type=float, default=0, metavar="T", help="sample the move at temperature T instead of argmax (0 = argmax)")
    ap.add_argument("--trace", type=int, default=10, help="decisions to show before each death")
    ap.add_argument("--flap-at", type=float, default=0.5, help="flap when p(flap) is at least this (default 0.5 = argmax)")
    ap.add_argument("--style", choices=list(STYLES), help="with --state semantic: what the options say (position, actions, actions plus the rule)")
    ap.add_argument("--noun", default="option", help="what the options are called in the prompt (option, statement, action)")
    ap.add_argument("--model", default="models/Qwen3.5-4B-Q4_K_M.gguf")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--watch", action="store_true", help="draw the game in the terminal")
    ap.add_argument("--delay", type=float, default=0.05)
    ap.add_argument("--agree", type=int, default=0, metavar="N", help="agreement with the oracle on N states instead of playing")
    args = ap.parse_args()
    TRACE[0] = args.trace
    DROP.update(x for x in args.drop.split(",") if x)
    policy = make_policy(args.policy, args, random.Random(args.seed))

    if args.agree:
        report_agreement(f"{args.policy}/{args.options}", *agreement(policy, args.agree))
        return
    res = []
    for i in range(args.episodes):
        r = play(args.seed + i, policy, args.max_steps, args.watch, args.delay)
        res.append(r)
        print(f"episode {i}: score {r['score']:2d}  steps {r['steps']:4d}  flaps when below the centre "
              f"{r['flap_recall']:.2f} (n={r['n_flap']})  {r['lat_ms']:.0f} ms/decision", flush=True)
    sc = [r["score"] for r in res]
    report_errors(res)
    print(f"{args.policy}/{args.options}: score mean {np.mean(sc):.2f} median {np.median(sc):.1f} max {max(sc)}  "
          f"(cap {args.max_steps // 30 - 2} at {args.max_steps} steps)")


if __name__ == "__main__":
    main()
