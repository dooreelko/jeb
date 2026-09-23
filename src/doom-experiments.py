"""Doom (vizdoom defend_the_center) state-description ladder — see moth blbci for the decision
record and the variant 1 (mirror openjev's bucketed describe) result. src/doom.py is variant 1
on its own, kept short as the baseline; this file holds the rest of the ladder so they can be
compared with one flag.

Same env, actions, readout (src/jev.py) and 4-tics-per-decision cadence as src/doom.py — only
describe() changes between variants.

usage: scripts/doom-experiments.sh --variant {1,2,3,4,5,6} [--episodes N] [--max-steps M] [--model path.gguf]
       variant 4: a short free-text reasoning pass (see reason()) before the same letter readout,
       on top of variant 1's describe(). variant 5: one independent yes/no judgment per action
       (see per_action()), highest p(yes) wins. variant 6: the same without the letter scaffold,
       plain yes/no tokens (see direct_yes()); --question right|best.
"""
import argparse
import random

import vizdoom as vzd
from scipy.special import softmax

from .doom import ACTIONS, BUTTONS, SCREEN_H, SCREEN_W, TICS_PER_DECISION, enemies, make_game
from .jev import Jev


def describe_v1(state):
    """Mirrors openjev's render_text bucketing (same as src/doom.py): offset -> left/right/on,
    distance -> very close/close/far."""
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


def offset_bucket(off):
    """Finer than v1's 3-way left/right/on: 7 graduated buckets either side of the crosshair."""
    for edge, name in ((-0.2, "far left of"), (-0.08, "left of"), (-0.02, "slightly left of"),
                       (0.02, "exactly on"), (0.08, "slightly right of"), (0.2, "right of")):
        if off < edge:
            return name
    return "far right of"


def distance_bucket(size):
    """Finer than v1's 3-way very close/close/far: 5 graduated buckets on relative size."""
    for edge, name in ((0.08, "very far"), (0.2, "far"), (0.35, "medium distance"), (0.5, "close")):
        if size < edge:
            return name
    return "very close"


def describe_v2(state):
    """Same shape as v1 (bucketed words), but with finer-grained buckets: tests whether the
    saturation v1 showed is about the wording itself or just how coarse it is."""
    ammo = state.game_variables[0] if state else 0
    health = state.game_variables[1] if state else 0
    es = enemies(state)
    if not es:
        seen = "No enemies are visible right now."
    else:
        parts = [f"a {name} {offset_bucket(off)} the crosshair ({distance_bucket(size)})" for name, off, size in es]
        seen = "You see " + ", ".join(parts) + "."
    return f"Doom, defending the center. Ammo: {int(ammo)}. Health: {int(health)}. {seen}"


def describe_v3(state):
    """No hand-picked words at all: enemy list as plain numeric facts (offset, size), same ammo/
    health facts. The model gets a scene description and has to judge distance/direction itself."""
    ammo = state.game_variables[0] if state else 0
    health = state.game_variables[1] if state else 0
    es = enemies(state)
    if not es:
        seen = "No enemies are visible."
    else:
        parts = [f"a {name} at horizontal offset {off:+.2f} from the crosshair (screen widths, negative is left) "
                 f"and apparent size {size:.2f} (fraction of screen height)" for name, off, size in es]
        seen = "Enemies: " + "; ".join(parts) + "."
    return f"Doom, defending the center. Ammo {int(ammo)}, health {int(health)}. {seen}"


DESCRIBE = {1: describe_v1, 2: describe_v2, 3: describe_v3}

REASON_PROMPT = "In one short phrase, what should the player do right now and why?"


def reason(jev, context, max_tokens=24):
    """A short free-text plan before the letter readout (variant 4): one greedy generation,
    capped small for speed. Not openjev's two-step (that's N classifier passes, no free text) —
    this is the other two-step idea: think briefly, then read the same option-letter logits."""
    prompt = jev.model.chat(f"{context}\n\n{REASON_PROMPT}")
    return jev.model.generate(prompt, max_tokens=max_tokens).strip()


def per_action(jev, context):
    """Variant 5: one independent yes/no judgment per action, no rival options in the prompt,
    then the action with the highest p(yes). The decomposition openjev's scoring does (one
    judgment per hypothesis), done with jeb's letter readout. A constant bias towards the first
    letter hits every call alike, so it cancels in the argmax."""
    return [jev.probabilities(f'{context} Question: is "{a}" the right move right now?', ["Yes", "No"],
                              noun="answer")[0] for a in ACTIONS]


QUESTION = {"right": "right move right now", "best": "best move right now"}


def direct_yes(jev, context, question="right"):
    """Variant 6: variant 5's per-action judgment without the multiple-choice scaffold. Plain
    yes/no question, p(yes) read straight from the yes/no tokens (both cases, which carry ~99% of
    the mass at that position), no letters to map the answer onto."""
    m = jev.model
    yes, no = [m.tok(t) for t in ("yes", "Yes")], [m.tok(t) for t in ("no", "No")]
    out = []
    for a in ACTIONS:
        logits = m.last_logits(m.chat(f'{context} Is "{a}" the {QUESTION[question]}?', system="Answer only yes or no."))
        p = softmax(logits[yes + no])
        out.append(float(p[:len(yes)].sum()))
    return out


def baseline_yes(jev, episodes=3, base=50000):
    """Each action's mean p(yes) over states from random play on separate seeds, frozen before
    the scored episodes: the flat "yes" prior per action, subtracted before the argmax (as flappy's
    fitted threshold was). Keeps the combiner dumb: no rule, just a per-action offset."""
    rng, total, n = random.Random(base), [0.0] * len(ACTIONS), 0
    for i in range(episodes):
        game = make_game(base + i)
        while not game.is_episode_finished():
            for j, p in enumerate(per_action(jev, describe_v1(game.get_state()))):
                total[j] += p
            n += 1
            c = rng.randrange(len(ACTIONS))
            game.make_action([b == BUTTONS[c] for b in BUTTONS], TICS_PER_DECISION)
        game.close()
    return [t / n for t in total]


def run_control(args):
    kills = []
    for i in range(args.episodes):
        rng, game, steps = random.Random(i), make_game(args.seed + i), 0
        while not game.is_episode_finished() and steps < args.max_steps:
            c = 2 if args.control == "attack" else rng.randrange(len(ACTIONS))
            game.make_action([b == BUTTONS[c] for b in BUTTONS], TICS_PER_DECISION)
            steps += 1
        kills.append(int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)))
        game.close()
        print(f"episode {i}: kills {kills[-1]}  steps {steps}", flush=True)
    print(f"control {args.control}: mean kills {sum(kills) / len(kills):.2f}  (openjev baseline ~11)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, choices=[*DESCRIBE, 4, 5, 6])
    ap.add_argument("--question", choices=list(QUESTION), default="right", help="variant 6: ask for the right or the best move")
    ap.add_argument("--model", default="models/Qwen3.5-0.8B-Q4_K_M.gguf")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=2100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--watch", action="store_true", help="print state (and reasoning, for variant 4) each decision")
    ap.add_argument("--calibrate", action="store_true", help="variant 5: subtract each action's baseline p(yes), fitted on separate states")
    ap.add_argument("--control", choices=["random", "attack"], help="no model: a random or always-attack policy, for reference")
    args = ap.parse_args()
    if args.control:
        return run_control(args)
    if args.variant is None:
        ap.error("--variant or --control is required")
    jev = Jev(args.model)
    offset = [0.0] * len(ACTIONS)
    if args.variant == 5 and args.calibrate:
        offset = baseline_yes(jev)
        print("baseline p(yes) fitted on separate states: " + " ".join(f"{a}={o:.3f}" for a, o in zip(ACTIONS, offset)), flush=True)
    describe = DESCRIBE.get(args.variant)  # None for variant 4, handled below

    kills = []
    for i in range(args.episodes):
        game = make_game(args.seed + i)
        steps = 0
        while not game.is_episode_finished() and steps < args.max_steps:
            state = game.get_state()
            context = describe_v1(state)
            shown = context
            if args.variant == 4:
                plan = reason(jev, context)
                shown = f"{context} Plan: {plan}"
            elif describe is not None:
                shown = describe(state)
            if args.variant in (5, 6):
                p_yes = per_action(jev, context) if args.variant == 5 else direct_yes(jev, context, args.question)
                choice = max(range(len(ACTIONS)), key=lambda j: p_yes[j] - offset[j])
                shown = f"{context} p(yes) " + " ".join(f"{a}={p:.2f}" for a, p in zip(ACTIONS, p_yes))
            else:
                choice = jev.probabilities(shown, ACTIONS, noun="action").argmax()
            action = [b == BUTTONS[choice] for b in BUTTONS]
            game.make_action(action, TICS_PER_DECISION)
            steps += 1
            if args.watch:
                print(f"step {steps}: {shown} -> {ACTIONS[choice]}", flush=True)
        k = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        kills.append(k)
        game.close()
        print(f"episode {i}: kills {k}  steps {steps}", flush=True)
    print(f"variant {args.variant}: mean kills {sum(kills) / len(kills):.2f}  (openjev baseline ~11)")


if __name__ == "__main__":
    main()
