# Experiments

One file per experiment (one hypothesis each), numbered in the order they were run. Every file has
the same four sections, so they can later be read side by side for an overarching analysis:

- **Hypothesis**: what we expected, or the question, stated before looking at the data.
- **Setup**: task, models, prompt/readout, metric, controls; how to reproduce.
- **Data**: the numbers, with sample sizes.
- **Conclusion**: what the data supports, what it does not, and what it led to.

Moth issues stay the decision records (what was decided and why); these files are the lab notebook
(what was tested and what came out). One moth task can hold several experiments.

The overarching analysis and lessons learned across all experiments will live in
`LESSONS.md` (not written yet).

## Framing

- **jev**: the original, closed decision product (TypeSafe).
- **openjev**: an open alternative that approaches it architecturally (a trained NLI cross-encoder).
- **jeb** (this repo): how close plain llama.cpp and a vanilla model get to the other two, by reading
  next-token logits instead of generating.

## Index

| # | experiment | moth | outcome in one line |
|---|---|---|---|
| 01 | [Logit readout on ag_news](01-logit-readout-ag-news.md) | gdeic | MC readout ≈ generation in accuracy and latency; dataset caps accuracy |
| 02 | [Is "N" needed in the Y/N scorer](02-is-n-needed.md) | rgdnq | No: ~all mass on Y/N, Y alone ranks as well |
| 03 | [Many classes: DBpedia-14](03-dbpedia-14.md) | m4pwf, dcjaj | 96-97%, calibrated, confidence separates right from wrong |
| 04 | [Abstention as an explicit option](04-abstention.md) | xuhkj | "None of the above" option ≫ confidence threshold |
| 05 | [Flappy Bird, numeric state](05-flappy-numeric-state.md) | cjd4t | 0-8 pipes vs openjev's 28; wording moves the boundary |
| 06 | [Flappy Bird, semantic state](06-flappy-semantic-state.md) | cjd4t | 28/28, but the prompt states the decision |
| 07 | [NPU offload](07-npu-offload.md) | nom4q, q8kte | Works for isolated matmuls; dead end next to a GPU |
| 08 | [Doom: state wording and model size](08-doom-wording-and-size.md) | blbci | Wording irrelevant; 0.8B worse than random, 27B 5.8 kills |
| 09 | [Doom: reasoning pass before the readout](09-doom-reasoning-pass.md) | blbci | Helps neither size |
| 10 | [Doom: per-action yes/no (decomposition)](10-doom-per-action.md) | blbci | Lifts 0.8B above controls; stacks with size (27B 7.6) |
| 11 | [Doom: yes/no read from the yes/no tokens](11-doom-direct-yes-no.md) | blbci | Worse in play; confounded (see 12) |
| 12 | [Doom: readout agreement factorial](12-doom-readout-factorial.md) | blbci | 0.8B reads only "turn left"; "right move" wording collides |
| 13 | [Doom: symbolic scene language](13-doom-symbolic-scene.md) | blbci | 0.8B at chance; 27B passes info but inverts turns |
| 14 | [Doom: clock positions](14-doom-clock-positions.md) | blbci | 27B best readout yet, right direction; 0.8B inverts |
| 15 | [Doom: best readout in play (27B)](15-doom-best-readout-play.md) | blbci | 5.8 kills, no better than joint MC; harness ≠ play |
| 16 | [Doom: clock-position scene in play (27B)](16-doom-clock-play.md) | blbci | 6.2 kills; cleanest policy yet, capped by aim resolution |
| 17 | [Doom: brevity (radio-call) scene](17-doom-brevity-scene.md) | blbci | Same as prose at 27B; 0.8B no rescue, "No contact." pulls it to attack |

Considered and not run: quantising openjev's own model to GGUF (it is a backbone plus a trained
classification head; llama.cpp cannot run the head at all). Recorded in moth blbci.
