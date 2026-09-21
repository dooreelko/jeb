Add GPQA-diamond to the benchmark comparison against Jev, Terra and openjev.

## Why
It is in TypeSafe's chart and in the openjev report (Jev about 44%, Terra about 38%, openjev 27%, random 25%). It is the hardest of the suite.

## Blocker
The dataset is gated: it needs a Hugging Face login and acceptance of the dataset terms by the user. That cannot be done on their behalf. Everything else in the suite is open.

## Decision
Do it after the open benchmarks, once access exists. Same protocol as the rest of the suite.
