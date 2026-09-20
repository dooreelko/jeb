"""README-style scorer: one Y/N pass per option, then compare options."""
from functools import cache

import numpy as np
from scipy.special import log_softmax, logsumexp

from .common import SYSTEM, intro

VARIANTS = ["gap", "p_yn", "y_logit", "p_full"]


@cache
def _yn_ids(model):
    return [model.tok("Y"), model.tok("N")]


def score_yn(model, article, classes):
    """One pass per class. Returns ({variant: one score per class}, mass_yn per class).

    gap     = logit(Y) - logit(N)
    p_yn    = log p(Y) with the softmax taken over {Y, N} only
    y_logit = logit(Y) alone, no N involved
    p_full  = log p(Y) with the softmax over the whole vocabulary, no N token involved
    mass_yn = probability mass on {Y, N} in the full vocabulary (do the tokens even get used?)
    """
    opts = "\n".join(f"- {c}" for c in classes)
    y, n = _yn_ids(model)
    out = {v: [] for v in VARIANTS}
    mass = []
    for c in classes:
        prompt = model.chat(
            f'{intro(article)}{opts}\n\nIs "{c}" the most likely topic?',
            system=SYSTEM.format(answer='"Y" or "N"'),
        )
        lg = model.last_logits(prompt)
        z = logsumexp(lg)
        out["gap"].append(lg[y] - lg[n])
        out["p_yn"].append(log_softmax(lg[[y, n]])[0])
        out["y_logit"].append(lg[y])
        out["p_full"].append(lg[y] - z)
        mass.append(np.exp(logsumexp(lg[[y, n]]) - z))
    return {k: np.array(v) for k, v in out.items()}, np.array(mass)
