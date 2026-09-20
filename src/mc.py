"""Multiple-choice scorer: one pass, options labelled A, B, C, ..., read the label-token logits."""
import string
from functools import cache

from .common import SYSTEM, intro

ABSTAIN = "None of the above"


@cache
def _label_ids(model, n):
    return [model.tok(l) for l in string.ascii_uppercase[:n]]


def score_mc(model, article, classes, abstain=False):
    """One score per option (label-token logits): the classes, then ABSTAIN if abstain is set.
    Works for up to 26 options."""
    options = [*classes, ABSTAIN] if abstain else list(classes)
    assert len(options) <= 26, "one letter per option"
    labels = string.ascii_uppercase[: len(options)]
    opts = "\n".join(f"{l}. {c}" for l, c in zip(labels, options))
    quoted = ", ".join(f'"{l}"' for l in labels[:-1]) + f' or "{labels[-1]}"'
    prompt = model.chat(
        f"{intro(article)}{opts}\n\nWhich topic is the most likely one?",
        system=SYSTEM.format(answer=f"with the letter of one option, {quoted}"),
    )
    return model.last_logits(prompt)[_label_ids(model, len(options))]
