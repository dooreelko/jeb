"""Baseline: ordinary greedy generation, parsed back to a class index (-1 if unparseable)."""
from .common import CLASSES


def gen_baseline(model, article):
    out = model.llm.create_chat_completion(
        messages=[{
            "role": "user",
            "content": f"Article: {article}\n\nClassify the topic as one of: {', '.join(CLASSES)}. Reply with only the topic name.",
        }],
        max_tokens=8, temperature=0,
    )
    txt = out["choices"][0]["message"]["content"].strip().lower()
    for i, c in enumerate(CLASSES):
        if txt.startswith(c.lower()):
            return i
    return -1
