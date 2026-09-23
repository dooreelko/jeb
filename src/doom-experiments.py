"""Doom (vizdoom defend_the_center) state-description ladder — see moth blbci for the decision
record and the variant 1 (mirror openjev's bucketed describe) result. src/doom.py is variant 1
on its own, kept short as the baseline; this file holds the rest of the ladder so they can be
compared with one flag.

Same env, actions, readout (src/jev.py) and 4-tics-per-decision cadence as src/doom.py — only
describe() changes between variants.

usage: scripts/doom-experiments.sh --variant {1,2,3,4,5,6,7} [--episodes N] [--max-steps M] [--model path.gguf]
       variant 4: a short free-text reasoning pass (see reason()) before the same letter readout,
       on top of variant 1's describe(). variant 5: one independent yes/no judgment per action
       (see per_action()), highest p(yes) wins. variant 6: the same without the letter scaffold,
       plain yes/no tokens (see direct_yes()); --question right|best. variant 7: per action, the
       best readout from the agreement harness (bare instruction, quoted frame, yes/no tokens).
       --agree N: no play, score readouts on N labelled states per target (--coded, --clock).
"""
import argparse
import math
import random

import vizdoom as vzd
from scipy.special import softmax

from .doom import ACTIONS, BUTTONS, SCREEN_H, SCREEN_W, TICS_PER_DECISION, enemies, make_game, threats
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


def describe_free(state):
    """Decision-free (see moth blbci): raw per-object observables in the screen's own frame, no
    relation to the crosshair computed in code. The model has to compare positions itself."""
    ammo = state.game_variables[0] if state else 0
    health = state.game_variables[1] if state else 0
    es = enemies(state)
    seen = ("Enemies on screen: " + "; ".join(f"a {name} at x {off + 0.5:.2f}, height {size:.2f}" for name, off, size in es) + "."
            if es else "No enemies on screen.")
    return (f"Doom, defending the center. Ammo {int(ammo)}, health {int(health)}. Screen x runs from 0 (left edge) "
            f"to 1 (right edge); the crosshair is at x 0.50. {seen}")


DESCRIBE = {1: describe_v1, 2: describe_v2, 3: describe_v3}
CONTEXT = {"1": describe_v1, "free": describe_free, "clock": lambda st: clock_scene(scene(st))["text"]}

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


def baseline_yes(ask, episodes=3, base=50000):
    """Each action's mean p(yes) over states from random play on separate seeds, frozen before
    the scored episodes: the flat "yes" prior per action, subtracted before the argmax (as flappy's
    fitted threshold was). Keeps the combiner dumb: no rule, just a per-action offset."""
    rng, total, n = random.Random(base), [0.0] * len(ACTIONS), 0
    for i in range(episodes):
        game = make_game(base + i)
        while not game.is_episode_finished():
            for j, p in enumerate(ask(game.get_state())):
                total[j] += p
            n += 1
            c = rng.randrange(len(ACTIONS))
            game.make_action([b == BUTTONS[c] for b in BUTTONS], TICS_PER_DECISION)
        game.close()
    return [t / n for t in total]


# ----------------------------------------------------------------------------- readout agreement
# Why did variant 6 lose to variant 5? The two prompts differ in three ways at once; score every
# combination on fixed, labelled states instead of playing (like flappy's --agree).

def target(state):
    """The obviously right action, from raw geometry: attack when the crosshair is inside a monster's
    box, else turn towards the monster nearest the crosshair; "none" when nothing is visible."""
    ts = threats(state)
    if not ts:
        return "none"
    mid = SCREEN_W / 2
    if any(l.x <= mid <= l.x + l.width for l in ts):
        return "attack"
    near = min(ts, key=lambda l: abs(l.x + l.width / 2 - mid))
    return "turn left" if near.x + near.width / 2 < mid else "turn right"


def scene(state):
    """What the harness keeps of a state: openjev's wording plus the facts, so other renderings
    (the coded one) can be built from the same states."""
    return {"text": describe_v1(state), "ammo": int(state.game_variables[0]), "health": int(state.game_variables[1]),
            "enemies": enemies(state)}


def labelled_states(n_per, base=60000, max_episodes=300):
    """Up to n_per scenes for each target, from random play on separate seeds."""
    rng, got, ep = random.Random(base), {k: [] for k in (*ACTIONS, "none")}, 0
    while min(map(len, got.values())) < n_per and ep < max_episodes:
        game = make_game(base + ep)
        ep += 1
        while not game.is_episode_finished():
            state = game.get_state()
            k = target(state)
            if len(got[k]) < n_per:
                got[k].append(scene(state))
            c = rng.randrange(len(ACTIONS))
            game.make_action([b == BUTTONS[c] for b in BUTTONS], TICS_PER_DECISION)
        game.close()
    return got


def ask_yes(jev, sc, persona, wrap, readout):
    """p(yes) per action for one combination of the three ways variants 5 and 6 differ.
    persona: "You are a decision maker..." (v5) or a bare instruction (v6).
    wrap: the context quoted inside jeb's "Given a context of ..." frame (v5) or plain text (v6).
    readout: A. Yes / B. No letters (v5) or the yes/no tokens themselves (v6).
    (decision, quoted, letters) is variant 5 exactly; (bare, plain, tokens) is variant 6 exactly."""
    m, context = jev.model, sc["text"]
    letters = readout == "letters"
    ans = 'with the letter of one option, "A" or "B"' if letters else "yes or no"
    system = f"You are a decision maker. You can only answer {ans}." if persona == "decision" else \
        ('Answer only "A" or "B".' if letters else "Answer only yes or no.")
    yes, no = ([m.tok("A")], [m.tok("B")]) if letters else ([m.tok("yes"), m.tok("Yes")], [m.tok("no"), m.tok("No")])
    out = []
    for a in ACTIONS:
        if wrap == "quoted":
            user = f'Given a context of "{context} Question: is "{a}" the right move right now?"'
            user += " and possible answers of\nA. Yes\nB. No\n\nWhich answer is the most likely one?" if letters else ""
        else:
            user = f'{context} Is "{a}" the right move right now?' + ("\nA. Yes\nB. No" if letters else "")
        p = softmax(m.last_logits(m.chat(user, system=system))[yes + no])
        out.append(float(p[:len(yes)].sum()))
    return out


# abstract, symmetric symbols: no arrows, half-filled shapes, weapons, faces, or yes/no-looking marks
SYMBOLS = list("●■◆★⬢♣♠♥♦✿❖⬟⬣☘◉✦⌘♜☾⚘")


def ask_coded(jev, sc, rng):
    """The scene and the question in abstract symbols, glossed once in English; answer T (yes) or N
    (no). The scene shares no word with the actions, so matching "left of" to "turn left" on the
    surface is impossible: the model has to connect them through the glossary. Symbols are drawn
    afresh per state, so no one symbol's own bias sticks. Same information as openjev's wording."""
    m = jev.model
    enemy, left, on, right, far, close, very, empty, *acts = rng.sample(SYMBOLS, 8 + len(ACTIONS))
    gloss = (f"{enemy} enemy. {left} left of the crosshair, {on} on the crosshair, {right} right of the crosshair. "
             f"{far} far, {close} close, {very} very close. {empty} no enemies visible. "
             + ", ".join(f"{s} {a}" for s, a in zip(acts, ("shoot", "turn left", "turn right"))) + ".")
    lines = [f"{enemy} {left if off < -0.015 else right if off > 0.015 else on} {very if size > 0.45 else close if size > 0.25 else far}"
             for _, off, size in sc["enemies"]] or [empty]
    system = "You are a decision maker in a symbolic language. You answer only T for yes or N for no."
    yes, no = [m.tok("T")], [m.tok("N")]
    out = []
    for s in acts:
        user = (f"Game: Doom, defend the center. Health {sc['health']}, ammo {sc['ammo']}.\nSymbols: {gloss}\n"
                f"Scene:\n" + "\n".join(lines) + f"\n{s}?")
        p = softmax(m.last_logits(m.chat(user, system=system))[yes + no])
        out.append(float(p[0]))
    return out


FOV = 90  # vizdoom's default horizontal field of view, degrees


def clock(off, brief=False):
    """Screen offset (fraction of width from the centre) as a clock position in half-hour steps,
    12 o'clock straight ahead: the bearing a pilot would call, no shared word with the actions."""
    bearing = math.degrees(math.atan(2 * off * math.tan(math.radians(FOV / 2))))
    half = round(bearing / 15)  # half-hours of 15 degrees
    hour = (12 + half // 2 - 1) % 12 + 1
    if half % 2 == 0:
        return f"{hour} o'clock"
    return f"{hour}-{hour % 12 + 1} o'clock" if brief else f"between {hour} and {hour % 12 + 1} o'clock"


def clock_scene(sc, brevity=False):
    """openjev's wording with "left of / on / right of the crosshair" replaced by clock positions.
    brevity: the same information as a terse radio call ("Contact: Demon, 10-11 o'clock, far."), a
    recognised register, instead of prose."""
    if brevity:
        dist = lambda size: "very close" if size > 0.45 else "close" if size > 0.25 else "far"
        calls = [f"{name}, {clock(off, brief=True)}, {dist(size)}" for name, off, size in sc["enemies"]]
        seen = "Contact: " + ". ".join(calls) + "." if calls else "No contact."
        return {**sc, "text": f"Doom, defend the center. Ammo {sc['ammo']}, health {sc['health']}. {seen}"}
    parts = [f"a {name} at {clock(off)} ({'very close' if size > 0.45 else 'close' if size > 0.25 else 'far'})"
             for name, off, size in sc["enemies"]]
    seen = "You see " + ", ".join(parts) + "." if parts else "No enemies are visible right now."
    return {**sc, "text": f"Doom, defending the center. Ammo: {sc['ammo']}. Health: {sc['health']}. {seen}"}


def agree(args):
    import itertools

    import numpy as np
    from sklearn.metrics import roc_auc_score
    states = labelled_states(args.agree)
    print("labelled states: " + ", ".join(f"{k} {len(v)}" for k, v in states.items()), flush=True)
    jev = Jev(args.model)
    # for each action: states where it is right vs states where it is clearly wrong
    # (with nothing visible neither turn is wrong, so "none" only counts against attack)
    neg = {"attack": ["turn left", "turn right", "none"], "turn left": ["turn right", "attack"], "turn right": ["turn left", "attack"]}
    rng = random.Random(args.seed)
    rows = [(combo, lambda c, combo=combo: ask_yes(jev, c, *combo))
            for combo in itertools.product(("decision", "bare"), ("quoted", "plain"), ("letters", "tokens"))]
    if args.coded:  # the best combination from the factorial, next to the coded prompt
        rows = [r for r in rows if r[0] == ("bare", "quoted", "tokens")] + [(("coded", "", ""), lambda c: ask_coded(jev, c, rng))]
    if args.clock:  # the best combination, scene in clock positions instead of left/on/right of the crosshair
        rows = [(("brevity" if args.brevity else "clock", "quoted", "tokens"),
                 lambda c: ask_yes(jev, clock_scene(c, args.brevity), "bare", "quoted", "tokens"))]
    print("persona  wrap    readout  | AUROC left right attack  mean | mean p(yes) left right attack | acc on/left/right  attack when none")
    for (persona, wrap, readout), ask in rows:
        p = {k: np.array([ask(c) for c in v]) for k, v in states.items()}
        aucs = []
        for j, a in enumerate(ACTIONS):
            pos, ng = p[a][:, j], np.concatenate([p[k][:, j] for k in neg[a]])
            aucs.append(roc_auc_score([1] * len(pos) + [0] * len(ng), np.r_[pos, ng]))
        allp = np.concatenate(list(p.values()))
        acc = np.mean([np.mean(p[a].argmax(1) == j) for j, a in enumerate(ACTIONS)])
        waste = np.mean(p["none"].argmax(1) == ACTIONS.index("attack"))
        tag = "  = v5" if (persona, wrap, readout) == ("decision", "quoted", "letters") else \
            "  = v6" if (persona, wrap, readout) == ("bare", "plain", "tokens") else ""
        print(f"{persona:8} {wrap:7} {readout:8} | {aucs[0]:.3f} {aucs[1]:.3f} {aucs[2]:.3f}  {np.mean(aucs):.3f} | "
              f"{allp[:, 0].mean():.2f} {allp[:, 1].mean():.2f} {allp[:, 2].mean():.2f} | {acc:.3f}  {waste:.2f}{tag}", flush=True)


# ----------------------------------------------------------------------------- recording
def captioned(game, step, action, p_yes, scale=1):
    """The current frame, upscaled, with a caption bar: what was decided and why."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(game.get_state().screen_buffer.transpose(1, 2, 0))  # CRCGCB -> HxWx3
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    out = Image.new("RGB", (img.width, img.height + 28), "black")
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
    ammo, health = (int(v) for v in game.get_state().game_variables)  # AMMO2, HEALTH per the cfg
    d.text((4, img.height + 2), f"step {step}  kills {kills}  health {health}  ammo {ammo}", fill="white")
    why = "" if p_yes is None else "yes: " + " ".join(f"{a.split()[-1]} {p:.2f}" for a, p in zip(ACTIONS, p_yes)) + "  "
    d.text((4, img.height + 15), f"{why}> {action.upper()}", fill="yellow")
    return out


def write_gif(frames, path, fps=35, keep_every=4, colors=64):
    """Palette-optimised gif through ffmpeg. Doom's noisy textures compress badly, so native size,
    every fourth tic (8.75 fps) and 64 colours keep a full episode to a few MB."""
    import subprocess
    w, h = frames[0].size
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps / keep_every), "-i", "-",
           "-vf", f"split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle", "-loop", "0", path]
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames[::keep_every]), check=True)


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
    ap.add_argument("--variant", type=int, choices=[*DESCRIBE, 4, 5, 6, 7])
    ap.add_argument("--context", choices=list(CONTEXT), default="1", help="variants 4-7: the state text the questions are asked about (1 = openjev's wording, free = absolute screen x, clock = clock positions)")
    ap.add_argument("--question", choices=list(QUESTION), default="right", help="variant 6: ask for the right or the best move")
    ap.add_argument("--model", default="models/Qwen3.5-0.8B-Q4_K_M.gguf")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=2100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--watch", action="store_true", help="print state (and reasoning, for variant 4) each decision")
    ap.add_argument("--calibrate", action="store_true", help="variant 5: subtract each action's baseline p(yes), fitted on separate states")
    ap.add_argument("--gif", metavar="PATH", help="record the first episode as a gif with a caption bar (decision and p(yes))")
    ap.add_argument("--control", choices=["random", "attack"], help="no model: a random or always-attack policy, for reference")
    ap.add_argument("--brevity", action="store_true", help="with --clock: the scene as a terse radio call instead of prose")
    ap.add_argument("--clock", action="store_true", help="with --agree: only the best combination with the scene in clock positions")
    ap.add_argument("--coded", action="store_true", help="with --agree: only the best combination and the coded (symbol) prompt")
    ap.add_argument("--agree", type=int, default=0, metavar="N", help="no play: score every persona/wrap/readout combination on N labelled states per target")
    args = ap.parse_args()
    if args.control:
        return run_control(args)
    if args.agree:
        return agree(args)
    if args.variant is None:
        ap.error("--variant, --control or --agree is required")
    jev = Jev(args.model)
    offset = [0.0] * len(ACTIONS)
    score = {5: lambda st: per_action(jev, CONTEXT[args.context](st)),
             6: lambda st: direct_yes(jev, CONTEXT[args.context](st), args.question),
             7: lambda st: ask_yes(jev, {"text": CONTEXT[args.context](st)}, "bare", "quoted", "tokens")}.get(args.variant)
    if score and args.calibrate:
        offset = baseline_yes(score)
        print("baseline p(yes) fitted on separate states: " + " ".join(f"{a}={o:.3f}" for a, o in zip(ACTIONS, offset)), flush=True)
    describe = DESCRIBE.get(args.variant)  # None for variant 4, handled below

    kills = []
    frames = [] if args.gif else None
    for i in range(args.episodes):
        game = make_game(args.seed + i)
        steps = 0
        while not game.is_episode_finished() and steps < args.max_steps:
            state = game.get_state()
            context = CONTEXT[args.context](state)
            shown = context
            if args.variant == 4:
                plan = reason(jev, context)
                shown = f"{context} Plan: {plan}"
            elif describe is not None:
                shown = describe(state)
            if score:
                p_yes = score(state)
                choice = max(range(len(ACTIONS)), key=lambda j: p_yes[j] - offset[j])
                shown = f"{context} p(yes) " + " ".join(f"{a}={p:.2f}" for a, p in zip(ACTIONS, p_yes))
            else:
                choice = jev.probabilities(shown, ACTIONS, noun="action").argmax()
            action = [b == BUTTONS[choice] for b in BUTTONS]
            if frames is None:
                game.make_action(action, TICS_PER_DECISION)
            else:  # same dynamics, one tic at a time so every frame is kept
                game.set_action(action)
                for _ in range(TICS_PER_DECISION):
                    game.advance_action(1)
                    if not game.is_episode_finished():
                        frames.append(captioned(game, steps + 1, ACTIONS[choice], p_yes if score else None))
            steps += 1
            if args.watch:
                print(f"step {steps}: {shown} -> {ACTIONS[choice]}", flush=True)
        k = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        kills.append(k)
        game.close()
        print(f"episode {i}: kills {k}  steps {steps}", flush=True)
        if frames is not None:
            write_gif(frames, args.gif)
            print(f"gif: {len(frames)} frames -> {args.gif}", flush=True)
            frames = None  # only the first episode is recorded
    print(f"variant {args.variant}: mean kills {sum(kills) / len(kills):.2f}  (openjev baseline ~11)")


if __name__ == "__main__":
    main()
