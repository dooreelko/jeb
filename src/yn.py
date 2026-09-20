"""README-style scorer: one Y/N pass per option, then compare options."""
from functools import cache

import numpy as np
from scipy.special import log_softmax

from .common import CLASSES

SYSTEM = 'You are a decision maker. You can only answer "Y" or "N".'


@cache
def _yn_ids(model):
    return [model.tok("Y"), model.tok("N")]


def score_yn(model, article):
    """N passes (one per class). Returns (gap, log_p_yes), each one score per class:
    gap = logit(Y) - logit(N);  log_p_yes = log softmax over {Y, N} at Y."""
    opts = "\n".join(f"- {c}" for c in CLASSES)
    yn = _yn_ids(model)
    gaps, log_pys = [], []
    for c in CLASSES:
        prompt = model.chat(
            f'Given a context of "{article}" and possible topics of\n{opts}\n\nIs "{c}" the most likely topic?',
            system=SYSTEM,
        )
        y_n = model.last_logits(prompt)[yn]
        gaps.append(y_n[0] - y_n[1])
        log_pys.append(log_softmax(y_n)[0])
    return np.array(gaps), np.array(log_pys)
