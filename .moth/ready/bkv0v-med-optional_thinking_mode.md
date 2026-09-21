Investigate an optional thinking mode: let the model reason in text before the readout, so the decision is still a probability over the option letters but comes after generated reasoning instead of straight after the prompt.

## Why
So far the readout is a single forward pass with thinking switched off (the empty reasoning block is prefilled). That is the fast, deterministic "System One" case Jev targets. In Flappy Bird it is also the limit: with the state as named buckets the model reads a stated comparison perfectly (28 pipes), but given only "Flap" / "Do nothing" plus a description of the game it cannot infer the move (score 0). A thinking model could likely take that small reasoning step. The question is what it costs and whether the answer distribution is still usable.

## Decision
- Thinking is an option, off by default, so the single-pass path and its numbers are unchanged.
- With thinking on, the model generates its reasoning, then the answer letter is read from the logits at the end as a softmax over the option letters, exactly as now. That keeps a proper distribution and the abstention option.
- Measure what thinking buys and what it costs, on the same tasks as before: Flappy Bird action wording (the case that currently scores 0), AG News and DBpedia accuracy and calibration.
- Report latency and tokens per decision next to accuracy and calibration. The trade is a few hundred tokens per decision against one pass, and the calibration of the letter distribution after reasoning is unknown.
- Bound the reasoning length so cost is predictable, and note what happens when it is cut off.

## Out of scope
Fine-tuning, real-time play, new benchmarks, and changing the default single-pass readout.

## Questions to answer
- Does thinking rescue the action wording in Flappy Bird, and how many pipes?
- Is the letter distribution after thinking still calibrated, or does it become overconfident?
- Does the same model in thinking mode remain worth it against greedy generation on latency?
