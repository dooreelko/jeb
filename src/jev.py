"""Front door shaped like openjev's OpenJevCrossEncoder: rerank(question, options), probabilities(...).

One forward pass per call, whatever the number of options. Nothing here is trained.
"""
import numpy as np
from scipy.special import softmax

from .common import Model
from .mc import score_mc


class Jev:
    def __init__(self, path, **model_kwargs):
        self.model = Model(path, **model_kwargs)

    def probabilities(self, context, options, abstain=False, noun="option"):
        """Distribution over the options; with abstain, one extra last entry for "None of the above"."""
        return softmax(score_mc(self.model, context, list(options), abstain=abstain, noun=noun))

    def rerank(self, question, options, abstain=False, noun="option"):
        """Index of the most likely option; len(options) means abstain."""
        return int(np.argmax(self.probabilities(question, options, abstain, noun)))
