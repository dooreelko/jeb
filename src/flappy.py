"""Turn-based Flappy Bird played by a one-pass multiple-choice readout: the winning setup (see flappy.md).

Every tick, code turns the game state into two named buckets (where the bird is against the gap centre, and
how it is moving), the model reads the logits of the option letters, and the argmax picks the move.
The experiments that led here are in flappy-experiments.py.

usage: scripts/flappy.sh [--no-watch] [--episodes N] [--max-steps M] [--seed S] [--model path.gguf]
"""
import argparse
import random

from .jev import Jev

GRAVITY, FLAP_V, SPEED = 0.003, 0.015, 0.015          # openjev's physics
PIPE_EVERY, GAP, PIPE_HW, BIRD_X, BIRD_HW = 0.45, 0.28, 0.04, 0.2, 0.03


class Flappy:
    def __init__(self, seed, max_steps):
        self.rng, self.max_steps = random.Random(seed), max_steps
        self.y, self.vy, self.t, self.score, self.done = 0.5, 0.0, 0, 0, False
        self.pipes = [[1.2, self.rng.uniform(0.15, 0.85 - GAP)]]  # [x, bottom of the gap]

    def next_pipe(self):
        return next((x, lo) for x, lo in self.pipes if x + PIPE_HW >= BIRD_X - BIRD_HW)

    def step(self, flap):
        self.vy = FLAP_V if flap else self.vy - GRAVITY
        self.y += self.vy
        self.t += 1
        for p in self.pipes:
            p[0] -= SPEED
        if self.pipes[-1][0] < 1.2 - PIPE_EVERY:
            self.pipes.append([self.pipes[-1][0] + PIPE_EVERY, self.rng.uniform(0.15, 0.85 - GAP)])
        if self.pipes[0][0] + PIPE_HW < BIRD_X - BIRD_HW:
            self.pipes.pop(0)
            self.score += 1
        x, lo = self.next_pipe()
        hit = self.y <= 0 or self.y >= 1 or (abs(x - BIRD_X) < PIPE_HW + BIRD_HW and not lo < self.y < lo + GAP)
        self.done = hit or self.t >= self.max_steps


def describe(env):
    """The state as words, no numbers: the bird against the gap centre, and its vertical motion."""
    d = env.y - (env.next_pipe()[1] + GAP / 2)
    where = next((name for edge, name in ((-0.15, "far below"), (-0.05, "below"), (0, "slightly below"),
                                          (0.05, "slightly above"), (0.15, "above")) if d < edge), "far above")
    v = env.vy
    motion = ("rising fast" if v > 0.006 else "rising slowly" if v > 0.0015 else "hovering" if v > -0.0015
              else "falling slowly" if v > -0.006 else "falling fast")
    return f"Flappy Bird. The bird is {where} the centre of the gap and is {motion}."


OPTIONS = ["The bird is below the centre of the gap", "The bird is above the centre of the gap"]  # A means flap


def draw(env, flap=False, w=60, h=18):
    grid = [[" "] * w for _ in range(h)]
    for x, lo in env.pipes:
        for c in range(max(int(x * (w - 1)) - 1, 0), min(int(x * (w - 1)) + 2, w)):
            for r in range(h):
                if not lo < 1 - r / (h - 1) < lo + GAP:
                    grid[r][c] = "#"
    grid[min(max(round((1 - env.y) * (h - 1)), 0), h - 1)][int(BIRD_X * (w - 1))] = "^" if flap else ">"
    return "\n".join([f"score {env.score}  step {env.t}   {'FLAP' if flap else '----'}", "=" * w, *("".join(r) for r in grid), "=" * w])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-0.8B-Q4_K_M.gguf")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--watch", action=argparse.BooleanOptionalAction, default=True, help="draw the game in the terminal (default; --no-watch to turn off)")
    args = ap.parse_args()
    jev = Jev(args.model)

    scores = []
    for i in range(args.episodes):
        env = Flappy(args.seed + i, args.max_steps)
        while not env.done:
            flap = jev.probabilities(describe(env), OPTIONS, noun="statement").argmax() == 0
            env.step(flap)
            if args.watch:
                print("\033[H\033[J" + draw(env, flap), flush=True)
        scores.append(env.score)
        print(f"episode {i}: score {env.score}  steps {env.t}", flush=True)
    print(f"mean score {sum(scores) / len(scores):.2f}  (cap {args.max_steps // 30 - 2} at {args.max_steps} steps)")


if __name__ == "__main__":
    main()
