import pytest

from syntaxbench.baselines import (
    MajorityBaseline, NearestNounBaseline, coverage, nearest_noun,
)
from syntaxbench.validation import parse_record


def test_majority_train_only():
    train = [parse_record(row) for row in ["VBP\tdogs", "VBP\tcats", "VBZ\tdog"]]
    model = MajorityBaseline.fit(train)
    held_out = [parse_record("VBZ\ta cat")] * 10
    assert [model.predict(text) for _, text in held_out] == [1] * 10
    assert MajorityBaseline.fit(reversed(train)) == model


def test_majority_tie_and_invalid():
    assert MajorityBaseline.fit([(1, "dogs"), (0, "dog")]).label == 0
    with pytest.raises(ValueError):
        MajorityBaseline.fit([])
    with pytest.raises(ValueError):
        MajorityBaseline.fit([(2, "dog")])


@pytest.mark.parametrize("tokens,expected", [
    ([("dog", "NN")], 0),
    ([("dogs", "NNS")], 1),
    ([("Alice", "NNP")], 0),
    ([("Alps", "NNPS")], 1),
    ([("dog", "NN"), ("near", "IN"), ("cats", "NNS")], 1),
    ([("dogs", "NNS"), ("quickly", "RB")], 1),
    ([("they", "PRP")], None),
    ([("dog", "NN"), ("unknown", "NN-UNKNOWN")], None),
    ([], None),
])
def test_nearest_noun(tokens, expected):
    assert nearest_noun(tokens) == expected


def test_coverage():
    assert coverage([0, None, 1, None]) == {
        "total": 4, "answered": 2, "abstained": 2, "coverage": 0.5,
    }
    assert coverage([])["coverage"] is None
    with pytest.raises(ValueError):
        coverage([2])


def test_preamble_only_adapter():
    inputs = []

    def tagger(text):
        inputs.append(text)
        return [("dogs", "NNS"), ("near", "IN"), ("tree", "NN")]

    model = NearestNounBaseline(tagger, "fixture-v1")
    assert model.predict("dogs near tree") == 0
    assert model.predict(" ") is None
    assert inputs == ["dogs near tree"]
    assert model.tagger_id == "fixture-v1"
    with pytest.raises(ValueError):
        NearestNounBaseline(tagger, "")
    with pytest.raises(TypeError):
        model.predict(None)
