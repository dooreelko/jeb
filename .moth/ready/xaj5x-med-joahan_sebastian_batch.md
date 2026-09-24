Per-action scoring (doom variants 5/6, see blbci) asks the model one short question per
candidate action. Today each question is a full, serial forward pass from scratch, so
N actions cost N full passes. On the 27B model that made runs impractically slow.

The questions are identical except for the action named near the end: the shared prefix
(system prompt + scene description + question start) is ~90% of the tokens.

## Goal

Score all candidate actions of one decision in roughly the cost of one pass, without
changing any decision: same prompts, same logits, same choices as the serial version.

## Approach (decided)

- Evaluate the shared prefix once, then reuse its cached state for every candidate
  instead of recomputing it.
- Evaluate the short per-candidate tails together, as parallel sequences in one batch,
  rather than one after another.
- Stay on vanilla llama.cpp. The high-level Python wrapper we use is built around a
  single sequence, so this needs its lower-level interface (multi-sequence batches,
  cache copy between sequences).
- A generic helper for "one context, N short continuations, read the answer logits of
  each", usable by doom and flappy alike, not doom-specific code.

## Why

- openjev does the equivalent by batching its premise/hypothesis pairs through the
  cross-encoder in one call. Batching alone gets parallelism; prefix reuse also removes
  the repeated premise compute, which a standard cross-encoder batch recomputes per row.
- Makes the per-action approach (the one that beat the joint multiple-choice readout)
  affordable on big models and closer to real time.

## Success criteria

- Identical choices to the serial version on the same seeds (equivalence check first).
- Per-decision latency for 3 actions close to one serial pass, measured on 0.8B and 27B.

## Out of scope

- Any change to prompts, wording or decision logic: this is speed only.
- Running a server / parallel slots: stays in-process.
