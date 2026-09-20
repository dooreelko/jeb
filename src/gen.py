"""Baseline: ordinary greedy generation, parsed back to a class index (-1 if unparseable)."""
from .common import CLASSES, intro


def gen_baseline(model, article):
    opts = "\n".join(f"- {c}" for c in CLASSES)
    prompt = model.chat(
        f"{intro(article)}{opts}\n\nWhich topic is the most likely one? Reply with only the topic name.",
        system="You are a decision maker.",
    )
    txt = model.generate(prompt, max_tokens=8).strip().lower()
    for i, c in enumerate(CLASSES):
        if txt.startswith(c.lower()):
            return i
    return -1
