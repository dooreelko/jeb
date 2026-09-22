we do vizdoom next. openjev claims 11 kills per episode with 1 decision per 4 tics.
we don't need realtime, but we can do 1 decision per 4 tics

flappy bird got saturated with decisions biased in the bucketed prompt.
here we explore what can we do without telling the engine what to decide.

ref: https://huggingface.co/AlexWortega/openjev/blob/main/code/doom.py

---

## Decision record

**Scenario**: vizdoom `defend_the_center` (same cfg openjev uses), so kills/episode is
directly comparable to their ~11 baseline. 1 decision per 4 tics (~114ms game time),
not realtime.

**Actions**: 3, matching openjev — turn left / turn right / attack.

**Readout**: reuse `src/jev.py`'s `Jev` (single-pass option-logit readout) unchanged.
Not openjev's NLI cross-encoder / fine-tuned MLP path.

**State description — the actual open question**: openjev's own `render_text()` buckets
enemy offset (left/right/on, threshold ±0.015) and distance (very close/close/far,
threshold on relative size) into words — same trap flappy hit (bucketed prompts biased
the decision, saturating scores). Explore as a ladder of variants, not one final design:

1. mirror openjev's bucketed describe() as baseline — confirms comparable kills/episode
   before diverging.
2. finer-grained buckets — same idea, less coarse, see if bias shrinks.
3. raw numeric state, no hand-picked words (enemy list with numeric offset/size, ammo,
   health, kills as plain facts) — still single-pass letter readout. Goal state: model
   judges purely from a scene description + option list, no baked-in directional bias.
4. if (3) degrades badly: add a free-text reasoning pass before the same single-pass
   action-letter readout — first point this goes beyond one forward pass. Open question
   for when we get there: how exactly (second pass on top of jev, or something else).

**Files**: `src/doom.py` (vizdoom env wrapper + current describe(), shaped like
flappy.py) + `src/doom-experiments.py` (the ladder, shaped like flappy-experiments.py)
+ `scripts/doom.sh` / `scripts/doom-experiments.sh` (copy `_common.sh` pattern from
scripts/flappy.sh).

**Out of scope**: openjev's NLI cross-encoder / fine-tuned MLP, screen-buffer/vision
input, scenarios other than defend_the_center.

**Metric**: kills/episode + steps survived per variant, logged like flappy runs
(logs/, tee).

---

## Variant 1 result (mirror openjev's bucketed describe)

5 episodes, seed 1000+i, max-steps 2100 (well above what's needed):

```
episode 0: kills 0  steps 73
episode 1: kills 0  steps 66
episode 2: kills 0  steps 82
episode 3: kills 0  steps 81
episode 4: kills 0  steps 66
mean kills 0.00  (openjev baseline ~11)
```

0 kills every episode, dies (health hits 0) in ~70-80 decisions (~280-320 tics),
nowhere near the 2100-tic timeout. Raw probabilities show why: "turn left" (option A,
listed first) sits around 0.66-0.80 almost regardless of state or which enemy is
visible — the model barely ever picks "attack". Same letter/position saturation flappy
hit, worse here since it also means the bird (player) never fights back.

Also fixed along the way: the labels buffer includes self (DoomPlayer/Marine*) and
transient FX (Blood, BulletPuff) alongside real monsters — filtered those out of
enemies() (didn't change the result, bias dominates regardless).

Next: variant 2/3 (finer buckets / raw numeric state) to see if the position bias is a
property of the bucketed wording specifically, or of the single-pass letter-readout
itself regardless of how the state is described.
