"""Evaluation datasets: how to load one and turn its rows into (text, label, class names)."""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from datasets import load_dataset


@dataclass
class Dataset:
    hf: str
    split: str
    classes: list[str]  # index = label id; names as the model should read them
    text: Callable[[dict], str]


DATASETS = {
    "ag_news": Dataset(
        "fancyzhx/ag_news", "test",
        ["World news", "Sports", "Business", "Science and technology"],
        lambda r: r["text"],
    ),
    "dbpedia_14": Dataset(
        "fancyzhx/dbpedia_14", "test",
        ["Company", "Educational institution", "Artist", "Athlete", "Office holder",
         "Means of transportation", "Building", "Natural place", "Village", "Animal",
         "Plant", "Album", "Film", "Written work"],
        lambda r: r["content"].strip(),
    ),
}


def load(name, n, seed=0, max_chars=600):
    """n shuffled examples: (texts cut to max_chars, integer labels, class names)."""
    d = DATASETS[name]
    ds = load_dataset(d.hf, split=d.split).shuffle(seed=seed).select(range(n))
    return [d.text(r)[:max_chars] for r in ds], np.array(ds["label"]), d.classes


def hide_classes(classes, y, k, seed=0):
    """Pretend k classes do not exist, to get out-of-scope examples with known labels.

    Returns (visible class names, labels remapped onto them with -1 for hidden-class examples,
    ids of the hidden classes)."""
    hidden = sorted(int(i) for i in np.random.default_rng(seed).choice(len(classes), size=k, replace=False))
    visible = [i for i in range(len(classes)) if i not in hidden]
    remap = {old: new for new, old in enumerate(visible)}
    return [classes[i] for i in visible], np.array([remap.get(int(v), -1) for v in y]), hidden
