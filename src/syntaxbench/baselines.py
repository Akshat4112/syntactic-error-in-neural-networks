"""Deterministic agreement baselines over canonical binary labels."""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class MajorityBaseline:
    label: int

    @classmethod
    def fit(cls, train_records):
        """Fit only on caller-supplied training records; ties favor singular."""
        counts = Counter()
        for label, _ in train_records:
            if type(label) is not int or label not in (0, 1):
                raise ValueError("Expected canonical binary label")
            counts[label] += 1
        if not counts:
            raise ValueError("Empty training set")
        return cls(int(counts[1] > counts[0]))

    def predict(self, preamble):
        return self.label


def nearest_noun(tagged_preamble):
    """Predict from the rightmost Penn Treebank noun; abstain if absent/unknown.

    POS tags must come from an independent tagger on the preamble alone.
    No gold subject or target verb annotations may be supplied.
    """
    for _, tag in reversed(tuple(tagged_preamble)):
        if tag in ("NN", "NNP"):
            return 0
        if tag in ("NNS", "NNPS"):
            return 1
        if tag.startswith("NN"):
            return None
    return None


def coverage(predictions):
    """Return counts and coverage, retaining abstentions in the denominator."""
    total = answered = 0
    for prediction in predictions:
        if prediction is not None and (
            type(prediction) is not int or prediction not in (0, 1)
        ):
            raise ValueError("Expected binary prediction or None")
        total += 1
        answered += prediction is not None
    return {
        "total": total,
        "answered": answered,
        "abstained": total - answered,
        "coverage": answered / total if total else None,
    }


@dataclass(frozen=True)
class NearestNounBaseline:
    """Adapt an independently supplied, versioned preamble POS tagger."""

    tagger: object
    tagger_id: str

    def __post_init__(self):
        if not callable(self.tagger):
            raise TypeError("Expected a callable POS tagger")
        if not isinstance(self.tagger_id, str) or not self.tagger_id.strip():
            raise ValueError("Record tagger version and configuration")

    def predict(self, preamble):
        if not isinstance(preamble, str):
            raise TypeError("Expected a raw text preamble")
        if not preamble.strip():
            return None
        return nearest_noun(self.tagger(preamble))
