we do vizdoom next. openjev claims 11 kills per episode with 1 decision per 4 tics.
we don't need realtime, but we can do 1 decision per 4 tics

flappy bird got saturated with decisions biased in the bucketed prompt.
here we explore what can we do without telling the engine what to decide.

ref: https://huggingface.co/AlexWortega/openjev/blob/main/code/doom.py

----- AI agent updates -------

## Framing

- jev: the original closed/black-box decision product.
- openjev: an open alternative approaching jev architecturally (trained NLI cross-encoder).
- jeb (this repo): how close can plain llama.cpp + a vanilla model get to the other two.

Broader question behind this task: is a jev-style one-pass readout useful for decisions,
and are we expecting too much "intelligence" from a single call? Alternative worth
testing: decompose the decision into small narrow judgments and compose the action from
a quorum (closer to how brains / ensembles work).

## Setup decisions

- vizdoom `defend_the_center` (openjev's scenario) so kills/episode compares to their ~11.
- 3 actions as in openjev: turn left, turn right, attack. One decision per 4 tics, not realtime.
- State comes from the game's object labels (enemy type, horizontal offset from crosshair,
  apparent size) plus ammo/health. The player itself and transient effects (blood, bullet
  puffs) are not enemies and are filtered out.
- Readout: jeb's existing one-pass multiple-choice letter readout, unchanged.
- Kept as a baseline entry point plus an experiments entry point with a variant switch,
  mirroring how flappy is organised. Metric: kills/episode and decisions survived,
  same seeds across variants, runs logged.

## Variant ladder and results (5 episodes each, same seeds)

| variant | idea | 0.8B | 27B |
|---|---|---|---|
| 1 | mirror openjev's bucketed wording (left/right/on, very close/close/far) | 0.00 | 5.80 |
| 2 | finer buckets (7 offset, 5 distance) | 0.00 | — |
| 3 | raw numbers, no descriptive words | 0.00 | — |
| 4 | short free-text "plan" generated first, then the same readout | 0.00 | 5.40 |

Findings:
- 0.8B: all four variants give identical episodes step for step. The readout puts
  ~0.6-0.8 on the first option ("turn left") almost regardless of the prompt; attack
  rarely above ~0.13. Wording is not the variable — the joint multiple-choice readout
  saturates on position at this size.
- 0.8B with a reasoning pass: its own plan text is sensible ("aim at the Demon and fire")
  but the readout ignores it. Chaining a weak component onto itself adds no capability.
- 27B: reads the state (5.4-5.8 kills, survives longer), about half openjev's ~11.
  Reasoning pass adds latency, no gain.

## Rejected

- Quantizing openjev's own model to GGUF: it is a backbone plus a trained classification
  head (contradiction/entailment/neutral), not a causal LM. llama.cpp cannot run the head
  even unquantized, and jeb's letter readout has nothing to read from it. Dropped.
- Reasoning pass (variant 4) as the fix: no help at either size.

## Next: variant 5 — per-action independent judgments (quorum)

Observation: openjev's scoring is itself a decomposition — one independent
premise/hypothesis judgment per action, combined by softmax — whereas jeb asks one prompt
to rank all options jointly. Variant 5 does the jeb-native equivalent: ask the model a
narrow yes/no question per action, independently (no competing options in the prompt),
then pick the action with the highest "yes". The combiner stays dumb (argmax/softmax), so
no hand-tuned rule sneaks bias back in.

Success criterion: run on 0.8B. Any real improvement over 0 kills means decomposition,
not scale, is the lever — and we're onto something.

## Variant 5 result (0.8B) and controls

Controls, no model, same seeds: random 1.0 kills/episode, always-attack 1.6. So 0.8B on
variants 1-4 (0.0) was worse than random; 27B (5.4-5.8) is genuinely reading the state.

Variant 5 on 0.8B: 2.20 (1/3/2/4/1). Above both controls; small sample. The action
breakdown matters more than the mean: with an enemy left it turned left or attacked, never
right; with an enemy right it turned right or attacked, never left. Zero wrong-direction
turns, where variants 1-4 turned left regardless. Decomposing broke the position
saturation and exposed a real state-reading signal in the 0.8B model.

What limits it: a flat prior towards "yes" for attack (~0.6 even on an empty screen), so it
attacks ~87% of decisions and wastes ammo. Natural next step keeps the combiner dumb:
remove each action's baseline "yes" level (measured on separate states, then frozen, as
flappy's fitted threshold did) before the argmax. Also: more episodes for a firmer number.

## Variant 5, 20 episodes (0.8B, same seeds)

| config | mean kills |
|---|---|
| random | 1.30 |
| always-attack | 1.50 |
| variant 5 | 2.30 |
| variant 5 + per-action baseline subtracted | 2.10 |

Paired with always-attack on the same seeds, variant 5 is never worse in any episode
(+0.8 mean, roughly 3 standard errors). The decomposition gain is real, if small.

Calibration (subtract each action's mean p(yes) measured on separate random-play states)
did not help: the fitted baselines are close together (turn left 0.57, turn right 0.54,
attack 0.62), so the offset barely reorders choices; 2.10 vs 2.30 is noise. So the limit is
not the flat prior but how weak the per-action "yes" signal is at 0.8B. Rejected as a fix.

Controls (random, always-attack) are now part of the experiments entry point so the
comparison is reproducible.

## Variant 5 on 27B (partial run, stopped early)

Same seeds as the earlier 27B variant 1 run:

```
seed:      1000 1001 1002 1003 1004   mean
v1 27B:      5    6    6    7    5    5.8
v5 27B:      8    6    7   12    5    7.6
```

Never worse per seed, +1.8 mean, one episode at 12 (above openjev's ~11). Decomposition
and model size stack. Run was stopped after 7 of 10 episodes (later ones 5, 5, 5) to free
the GPU: the paired comparison had already answered the question.

## Variant 6: drop the letter scaffold

Variant 5 still asks the yes/no through the multiple-choice format (A. Yes / B. No, read the
letter). That adds an answer-to-letter mapping step, which is where 0.8B broke in variants
1-4. Variant 6 asks a plain yes/no question and reads the yes/no tokens directly (on 0.8B,
yes/Yes/no/No carry ~99% of the probability at that position). Wording ("right move" vs
"best move", which is comparative) is tested as a separate variable, not in the same run.
