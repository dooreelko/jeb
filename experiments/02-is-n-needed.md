# 02 Is "N" needed in the Y/N scorer?

2026-09-20 · moth rgdnq

## Hypothesis

Open question rather than a bet: N may remove per-prompt offsets that shift Y and N together, or the
shared prompt may already cancel them, making N dead weight.

## Setup

Same passes as 01 (ag_news, 4B/9B/27B), each option scored four ways at no extra cost: Y−N logit gap;
log p(Y) over {Y, N}; raw Y logit alone; log p(Y) over the whole vocabulary. Also recorded: the
probability mass on {Y, N} at all.

## Data

See the Y/N rows in 01. Accuracy over all 200 examples (standard error ≈ 2.3 points):

| model | Y/N gap | Y logit only |
|---|---|---|
| 4B | 0.83 | 0.85 |
| 9B | 0.87 | 0.85 |
| 27B | 0.86 | 0.85 |

Mass on {Y, N}: > 0.99, so the {Y,N}-normalised and full-vocabulary variants coincide.

## Conclusion

N is not needed for ranking: Y alone ranks about as well as the gap. The variants differ only in
calibration, which differs by model. Moot afterwards, since Y/N was dropped in favour of MC (04).
