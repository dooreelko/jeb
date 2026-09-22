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

---

## Variants 2 and 3: same bias, not the wording

Ran both full (5 episodes, seed 1000+i, max-steps 2100):

```
variant 2 (finer buckets):  steps 73 66 82 81 66  kills 0 0 0 0 0
variant 3 (raw numeric):    steps 73 66 82 81 66  kills 0 0 0 0 0
```

Identical to variant 1, episode-length to the tic, across all three. Checked raw
probabilities directly on variant 3 (plain numeric offsets/sizes, no words at all):
"turn left" still ~0.60-0.64, "attack" never above ~0.13, essentially unmoved by
what is actually in the prompt (enemy present or not, offset value, nothing changes
the ranking).

**Diagnosis**: this is not a state-description problem. The 3-option single-pass
letter-readout (src/jev.py) is strongly biased toward option A ("turn left", listed
first) for this action-choice framing, close to independent of content. Changing
describe() — the whole premise of variants 1-3 — cannot fix a bias that lives in the
readout/prompt-framing for actions, not in how the scene is described.

**Ladder paused here.** Variant 4 (reasoning pass before the readout) was the planned
next step if a *content* variant degraded; it might still be worth trying since a
free-text pass could break the A-bias, but that is a different hypothesis than the
ladder was built to test, so flagging before spending more runs on it. Also worth
checking whether the bias is doom-specific (3 actions, "action" noun) or shows up in
flappy's framing too (2 options) — flappy's own history suggests a milder version of
the same thing.

---

## Model size matters: 27B breaks the A-bias

Same variant 1 (bucketed describe), --model models/Qwen3.8-27B-UD-Q4_K_M.gguf, 5
episodes:

```
episode 0: kills 5  steps 105
episode 1: kills 6  steps 102
episode 2: kills 6  steps 126
episode 3: kills 7  steps 136
episode 4: kills 5  steps 120
mean kills 5.80  (openjev baseline ~11)
```

All runs above used the 0.8B model (script default) and got 0 kills, stuck near
"turn left" regardless of content. The 27B model gets 5.80 mean kills and survives
longer (102-136 vs 66-82 decisions) — clearly reading the state, not just saturating
on option A. So the letter-position bias diagnosed above is (at least largely) a
small-model capability problem, not an inherent flaw in the single-pass letter-readout
itself. Still ~half openjev's ~11, room to close with variant 2/3 (finer/raw state) on
the 27B model, or the reasoning-pass idea (variant 4).

---

## Considered and dropped: quantizing openjev's own model

openjev is not a causal LM: a Qwen3.5 backbone plus a separate trained NLI
classification head (custom modeling_openjev.py, mlp_heads_35b/), scored by
premise/hypothesis entailment via OpenJevCrossEncoder.rerank(), not next-token
logits. GGUF/llama.cpp targets causal LMs; converting would drop the trained head
and jev.py's letter-readout has nothing to read from a cross-encoder anyway. Not
worth the mismatch — dropped in favor of pushing the 27B result further.
