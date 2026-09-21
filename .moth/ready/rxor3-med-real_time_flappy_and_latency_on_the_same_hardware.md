Compare latency and real-time play of our readout against openjev, in a way that is meaningful.

## Why it needs care
In real-time mode the game keeps ticking while the model decides, so the score is coupled to decision latency. Their published latencies (about 55-140 ms) come from unknown hardware with specialised kernels, so absolute numbers cannot be compared with ours. A slow machine would lose the game regardless of how good the decisions are.

## Decision
- Prefer hardware-independent cost: forward passes and prompt tokens per decision.
- Measure both systems on the same machine if openjev can be run here; otherwise report ours only and say so.
- Report the highest game speed our policy sustains, as a sweep, instead of a single latency claim.
- Reduce our per-decision cost first by reusing the fixed part of the prompt between decisions (only the game state changes).
- Repeat timings, since the integrated GPU shares memory and thermals with other work.

## Depends on
The turn-based Flappy comparison (decision quality without latency).

## Out of scope
Doom, Minecraft and the pixel-input variants.
