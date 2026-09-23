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

## Variant 6 result (0.8B, 20 episodes): rejected

| config | mean kills | attack share |
|---|---|---|
| always-attack | 1.50 | 100% |
| variant 5 (yes/no through A/B letters) | 2.30 | ~87% |
| variant 6, "right move" (yes/no tokens) | 1.65 | 95% |
| variant 6, "best move" | 1.45 | 96% |

Reading the yes/no tokens directly gives "attack" an even larger built-in "yes" than the
letter form, so it collapses towards always-attack, and wrong-direction turns reappear.
CORRECTED below: variants 5 and 6 differ in three ways at once (persona, framing, readout), so this run alone does not show the letter form is better. "Best" (comparative) was
no better than "right". Variant 5 stays the per-action baseline.

## Decision: prompts must not contain the decision

The goal of this task, made explicit: the prompt describes the scene, the model makes the
decision. Working definition: only raw observables of each object in the scene's own
frame; no relation, comparison or bucket computed in code.

- Decision-bearing: "Demon left of the crosshair", "offset -0.3 from the crosshair",
  flappy's "slightly below the gap centre" or "+0.02 relative to the centre".
- Decision-free: "Demon at screen x 0.21, crosshair at x 0.50"; "bird height 0.42, gap
  from 0.30 to 0.58".

Consequence: variants 1, 2, 4, 5, 6 all used openjev's decision-bearing wording
("left of" nearly says "turn left", "exactly on" nearly says "attack"), so variant 5's
direction-consistent turns may be word-matching rather than scene reading. Variant 3 is
in between (offset relative to the crosshair). Results so far are about readouts, not
about decision-free play.

Next: variant 5 readout on a strictly decision-free doom description; and in flappy, a
cheap agreement/AUROC check of letter vs yes/no readouts on openjev's numeric state with
the decision lines dropped.

## Why did variant 6 lose? Readout agreement (0.8B)

Method: no play. Fixed states from random play on separate seeds, labelled from raw
geometry (crosshair inside a monster's box = attack, else turn towards the nearest one,
nothing visible = none), 40 per label. Every combination of the three ways variants 5
and 6 differ (persona: "decision maker" vs bare instruction; framing: context quoted in
jeb's "Given a context of" frame vs plain; readout: A/B letters vs yes/no tokens) is
scored per action: AUROC of p(yes) separating states where that action is right from
states where it is clearly wrong (signal), mean p(yes) (bias), argmax accuracy, and how
often it attacks on an empty screen.

Findings:
- Neither variant 5 nor 6 really reads the scene. Almost all signal is "turn left"
  (AUROC 0.64-0.81); "turn right" and "attack" sit at or below chance in nearly every
  combination. Variant 5 beat 6 in play mostly via a stronger turn-left signal (0.72 vs
  0.64) plus noise. No single factor explains it; they interact.
- Suspected cause of the left/right asymmetry: the question wording itself, "is \"turn
  right\" the right move right now?" uses "right" three times in two meanings, which
  flattens p(yes) for turn right. Turn left has no collision, so the scene's "left of"
  gets through. Present in variants 5 and 6 alike (and in "best move right now").
- Attack is capped by the context, not the readout: openjev's wording says "exactly on"
  only for a near-dead-centre enemy, so many crosshair-on-target states read "left of" /
  "right of".
- One combination reads all three actions (bare persona, quoted frame, yes/no tokens:
  AUROC 0.74/0.80/0.73) but its per-action levels are not comparable across actions
  (argmax accuracy 0.36), so it would need per-action calibration in play.

Next: reword the question to avoid "right" and rerun the factorial with more states;
then play the best combination with calibration.

## Symbolic scene language (the user's idea): does the model read or word-match?

Prompt: the scene and the question in abstract symmetric symbols (no arrows, faces,
weapons or yes/no-looking marks), glossed once in English, re-drawn per state so no one
symbol's bias sticks; answer T (yes) / N (no); one question per action. Same information as
openjev's wording, but the scene shares no word with the actions. (T/N hold ~80% of the
probability at the answer position on 0.8B; the rest is mostly a symbol.) Scored with the
agreement harness, 40 states per label, next to the best plain combination (bare persona,
quoted frame, yes/no tokens).

| model | prompt | AUROC turn left | turn right | attack | attacks on empty screen |
|---|---|---|---|---|---|
| 0.8B | best plain | 0.74 | 0.80 | 0.73 | 38% |
| 0.8B | coded | 0.53 | 0.50 | 0.50 | 28% |
| 27B | best plain | 0.90 | 0.87 | 0.71 | 0% |
| 27B | coded | 0.31 | 0.29 | 0.75 | 30% |

- 0.8B: chance on the coded prompt. Its signal was surface word matching.
- 27B: information does pass through the symbols (attack 0.75), but both turns are
  inverted (~3 standard errors below chance): "enemy left of the crosshair" maps to
  "turn right" (camera-pan confusion). So even 27B's correct turning in plain English is at
  least partly word matching; forced to reason spatially, it gets direction backwards.
- 27B with the best plain combination reads all three actions and never shoots at an
  empty screen: the strongest readout found so far, worth taking into play.

## Discussion: what "unbiased" can mean here

A fully allocentric scene (enemy and own coordinates plus heading) would need vector math
or trigonometry, which jev itself rules out for this kind of engine. And it is not more
unbiased, only less processed: the renderer already projects the world onto the screen,
as eyes do. The line that holds: perception-level quantities (bearing relative to gaze,
apparent size, health, ammo) are fair; naming or pre-drawing the decision is not. In
doom, perception and decision are one hop apart by the nature of the game. The bias that
matters is shared vocabulary between scene and actions. A natural-language middle ground
without math or shared words: clock positions ("enemy at 10 o'clock, close"), a convention
models know rather than one they must learn from the prompt.

## Clock positions and the best harness readout in play

Clock positions (perceptual bearing, no shared word with the actions, a convention models
already know), best English readout otherwise, same labelled states:
- 27B: AUROC turn left 0.94, turn right 0.92, attack 0.77: the best readout on any prompt,
  turning the right way. Its inversion on the symbolic scene came from a convention it had
  to learn from the prompt.
- 0.8B: turns inverted (0.34 / 0.35) with clock positions too; its correct English turning
  was word matching.

Best harness readout in play (27B, per action, calibrated, seeds 1000-1004): 5.8 kills,
equal to the joint multiple choice and below per-action letters (7.6). Harness separation
predicts play only loosely, as in flappy: per-action ranking across states is not the
per-state comparison between actions that play needs, and random-play states are not the
states a good policy visits.

Decision: every experiment is also summarised under experiments/ (hypothesis, setup,
data, conclusion), towards an overarching lessons-learned analysis; this task's
experiments are 08-15 there.
