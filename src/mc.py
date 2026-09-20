"""Multiple-choice scorer: one pass, options labelled A/B/C/D, read the label-token logits."""
from functools import cache

from .common import CLASSES

LABELS = ["A", "B", "C", "D"]


@cache
def _label_ids(model):
    return [model.tok(l) for l in LABELS]


def score_mc(model, article):
    """Returns one score per class (label-token logits)."""
    opts = "\n".join(f"{l}. {c}" for l, c in zip(LABELS, CLASSES))
    prompt = model.chat(
        f"Article: {article}\n\nWhich topic is this article about?\n{opts}\n\nAnswer with a single letter."
    )
    return model.last_logits(prompt)[_label_ids(model)]
