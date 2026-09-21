# Flappy Bird with the multiple-choice readout

Turn-based Flappy Bird with openjev's physics, played by a plain Qwen3.5 through the multiple-choice
readout (one pass, options `A`/`B`, argmax picks the move, option A = flap). The target is openjev's
27.5-28 pipes; a perfect rule scores 28 and a lookahead oracle about 25. Run it with `scripts/flappy.sh`
(`--watch` draws the game in the terminal). Decisions and numbers are also in moth `cjd4t`.

## What we learned

1. **The state format decides everything.** Numbers in the prompt gave at best 8.33 pipes. Replacing them
   with a state computed in code as named buckets ("The bird is far below the centre of the gap and is
   rising fast.") gave 28 pipes in all six episodes at plain argmax, with no tuning.
2. **This follows the Jev 1.13 docs** (docs.typesafe.ai/model-jaggedness/jev-1.13): semantic
   representations beat numeric ones ("pass in either the computed number or a named bucket"), unrelated
   state acts as a distractor ("send only the fields the question needs"), instructions should be direct,
   and score calibration is weak. Our earlier numeric prompts broke most of these.
3. **The model executes a stated comparison, it does not infer the move.** With options worded like the
   state ("The bird is below/above the centre of the gap") it scores 28. With "Flap" / "Do nothing" and
   only a description of the game it scores 0, and so does a goal-level line ("The bird wants to stay
   level with the centre of the gap"): it answers "flap" almost always. An explicit rule line brings 28
   back, but that leaks the policy.
4. **So the 28 is not "the model decides the game".** The mapping from statement to move ("below the
   centre means flap") is hand-written, and so are the bucket edges. And it is no longer a like-for-like
   comparison with openjev, whose model reads raw numbers.
5. **With numbers in the prompt**, prompt wording moved the score between 0 and about 8:
   - The signed offset from the gap centre is the essential input; raw heights alone are at chance.
   - openjev's goal sentence and the final period on the option shift the boundary toward "flap":
     verbatim prompt, balanced accuracy 0.76 and score 0; without the goal sentence, 0.89 and 2.33.
   - Every death was a near-tie flap (p 0.51-0.59) when the bird was already far above the gap and
     starting to fall: the model reacted to "falling" more than to position.
   - A 9B was no better than the 4B (5.83 against 8.33).
6. **Threshold and sampling.** A stricter flap cut-off (0.6) lifted the numeric variant from 3.33 to
   22.00, but that patches a bad boundary; it was not pursued (prompts are fixed first, tricks last).
   Sampling the move at temperature 0.5, 1 or 2 scored 0 every time: keep the readout an argmax.
7. **Hints in the numeric prompt backfire.** "The centre of the gap is above the bird" mirrors the option
   wording and collapsed the model to chance.

## Results (4B unless noted, 6 episodes, 900 steps, seeds 1000-1005)

Semantic state, plain argmax:

| options | reading, balanced acc at 0.5 | game score, mean |
|---|---|---|
| position statements | 0.99 | **28** (all six); also 28 in all six on fresh seeds 2000-2005 |
| actions, plain game description | 0.50 | 0 |
| actions plus the explicit rule (leaks the policy) | 0.99 | 28 (all six) |
| actions plus a goal line (fresh seeds 2000-2005) | 0.50 | 0 (all six die at step 34) |

Numeric state, position statements:

| state text | reading, balanced acc at 0.5 | game score, mean (max) |
|---|---|---|
| shortened copy of openjev's | 0.96 | 8.33 (18) |
| shortened, Qwen3.5 9B | - | 5.83 (9) |
| openjev's verbatim | 0.76 | 0 |
| verbatim minus goal sentence | 0.89 | 2.33 (5) |
| verbatim minus position and goal sentences, no final period | 0.90 | 3.33 (8) |
| same, flap at p >= 0.6 (tuning the boundary) | - | 22.00 (28) |
| same, sampled at T = 0.5 / 1 / 2 | - | 0 |
| words only (hints, ablation) | 0.86 | 0 |

References: never flap 0, random about 0.17, lookahead oracle about 25, perfect rule 28, openjev 27.5-28.
The reading column is measured on 100 states balanced between "below" and "above" the gap centre.

## Caveats

- 6 episodes per row on one model. The winning style also scores 28 in all six on fresh seeds
  (2000-2005) that were not used while choosing prompts; the other rows were not repeated on them.
- Reading accuracy on balanced states predicted the game only loosely (balanced accuracy 0.76, 0.89, 0.90,
  0.96 gave 0, 2.33, 3.33, 8.33; AUROC did not track it). The states come from a noisy oracle's play, not
  from the model's own, so they under-represent the drifted states where it dies.
- Plain agreement with an oracle is misleading (the oracle flaps about 10% of ticks, so never flapping
  scores about 0.90); we report balanced accuracy and AUROC against the truth of the statement.
- openjev's own run was played at 15 fps real time; ours is turn-based.
- The numeric prompts here are shortened variants of openjev's text, so they are ablations.

## Untried

Coarser or shifted bucket edges as a robustness check, action repeat, few-shot examples,
models beyond 9B.
