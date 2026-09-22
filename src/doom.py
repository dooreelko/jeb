"""Turn-based vizdoom (defend_the_center) played by a one-pass multiple-choice readout, same
readout as flappy.py (src/jev.py). describe() below is variant 1 of the ladder: mirrors
openjev's own bucketed render_text (see doom.md once written) as the comparable baseline
before we try to strip the bucketing out. See moth issue blbci for the decision record.

usage: scripts/doom.sh [--no-watch] [--episodes N] [--max-steps M] [--model path.gguf]
"""
import argparse

import vizdoom as vzd

from .jev import Jev

ACTIONS = ["turn left", "turn right", "attack"]  # index order must match BUTTONS below
BUTTONS = [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK]
TICS_PER_DECISION = 4  # not realtime; openjev's own cadence


def make_game(seed):
    game = vzd.DoomGame()
    game.load_config(vzd.scenarios_path + "/defend_the_center.cfg")
    game.set_window_visible(False)
    game.set_labels_buffer_enabled(True)  # for per-enemy screen position/size
    game.set_seed(seed)
    game.init()
    return game


SCREEN_W, SCREEN_H = 320, 240  # RES_320X240, set by defend_the_center.cfg


def enemies(state):
    """Visible enemies as (name, offset, size): offset in [-0.5, 0.5] of screen width from
    centre, size as fraction of screen height. Sorted by distance from the crosshair."""
    if state is None or state.labels is None:
        return []
    # defend_the_center's labels buffer includes self (DoomPlayer/Marine*) and transient FX
    # (Blood, BulletPuff) alongside the actual monsters; only the latter are threats.
    not_a_threat = ("DoomPlayer", "Marine", "Blood", "BulletPuff")
    out = [(l.object_name, l.x / SCREEN_W + l.width / (2 * SCREEN_W) - 0.5, l.height / SCREEN_H)
           for l in state.labels if not l.object_name.startswith(not_a_threat)]
    return sorted(out, key=lambda e: abs(e[1]))


def describe(state):
    """Variant 1: mirrors openjev's render_text bucketing (see moth blbci) as the baseline."""
    ammo = state.game_variables[0] if state else 0
    health = state.game_variables[1] if state else 0
    es = enemies(state)
    if not es:
        seen = "No enemies are visible right now."
    else:
        parts = []
        for name, off, size in es:
            side = "left of" if off < -0.015 else "right of" if off > 0.015 else "exactly on"
            dist = "very close" if size > 0.45 else "close" if size > 0.25 else "far"
            parts.append(f"a {name} {side} the crosshair ({dist})")
        seen = "You see " + ", ".join(parts) + "."
    return f"Doom, defending the center. Ammo: {int(ammo)}. Health: {int(health)}. {seen}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-0.8B-Q4_K_M.gguf")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--watch", action=argparse.BooleanOptionalAction, default=False, help="print state each decision (off by default; vizdoom has no terminal renderer here)")
    args = ap.parse_args()
    jev = Jev(args.model)

    kills = []
    for i in range(args.episodes):
        game = make_game(args.seed + i)
        steps = 0
        while not game.is_episode_finished() and steps < args.max_steps:
            state = game.get_state()
            choice = jev.probabilities(describe(state), ACTIONS, noun="action").argmax()
            action = [b == BUTTONS[choice] for b in BUTTONS]
            game.make_action(action, TICS_PER_DECISION)
            steps += 1
            if args.watch:
                print(f"step {steps}: {describe(state)} -> {ACTIONS[choice]}", flush=True)
        k = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        kills.append(k)
        game.close()
        print(f"episode {i}: kills {k}  steps {steps}", flush=True)
    print(f"mean kills {sum(kills) / len(kills):.2f}  (openjev baseline ~11)")


if __name__ == "__main__":
    main()
