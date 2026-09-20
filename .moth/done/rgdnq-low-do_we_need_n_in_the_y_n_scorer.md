Question: in the per-option Y/N scorer (see IDEA.md), is the "N" reading needed, or is the "Y" reading alone enough to rank options?

## Why it matters
The Y/N scorer costs one forward pass per option. Knowing which part of the reading carries the signal tells us whether the extra token is dead weight, and whether the Y/N scorer can be simplified or folded into a cheaper design. It also affects calibration, which is the main weakness of raw logit readout.

## What is known
- The two variants computed so far, the logit gap between Y and N and the log-probability of Y after a softmax over just {Y, N}, are monotonic in each other per option. They therefore rank options identically (both scored the same accuracy) and differ only in calibration. That comparison says nothing about whether N is needed.
- The saved scores only hold those two variants, so the Y-only case cannot be recovered from existing results.

## Decision
Answer it empirically from the same forward passes, at no extra inference cost, by scoring each option four ways and comparing them on accuracy, log loss and calibration error (raw and after temperature scaling):
- logit gap between Y and N (N acts as a per-prompt baseline)
- log-probability of Y over just {Y, N}
- raw logit of Y alone (no N)
- log-probability of Y over the whole vocabulary (no N token needed, but the model's other candidate answers count against it)

Also record how much probability mass sits on {Y, N} at all. If most of it sits on other tokens (for example "Yes"), the Y-only scores are measuring noise and the token choice itself is the problem, not the normalization.

## Why not decide from theory
Argument in both directions is plausible: N removes offsets that shift Y and N together, but a shared prompt may already cancel most of them, and long option names could shift the offset per option. Only a measurement separates the two.

## Out of scope
Trying other answer tokens or wording, changing the scoring across options (the softmax across options stays), and any fine-tuning. Prompt alignment between the two scorers and saving latency are tracked separately in the Bootstrap task; this shares its rerun, not its scope.
