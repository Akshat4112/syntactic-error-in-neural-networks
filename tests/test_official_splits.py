import json

import pandas as pd
import pytest

from syntaxbench.dataset import dataset, fit_training_texts, load_split


def test_official_partitions(tmp_path):
    raw = tmp_path / "data" / "rnn_agr_simple"
    raw.mkdir(parents=True)
    (raw / "numpred.train").write_text("VBZ\ttrain only\nVBP\tshared\n")
    (raw / "numpred.val").write_text("VBP\tvalidation only\nVBZ\tshared\n")
    (raw / "numpred.test").write_text("VBZ\ttest only\n")
    dataset(tmp_path).preprocess_data()
    for name, expected in (("train", "train only"),
                           ("validation", "validation only"),
                           ("test", "test only")):
        frame = pd.read_csv(tmp_path / "data" / f"{name}_df.csv")
        assert frame.iloc[0]["text"] == expected
    report = json.loads((tmp_path / "data" / "split-overlap.json").read_text())
    assert report["unique_exact_preamble_overlap"]["train:validation"] == 1
    assert report["unique_exact_preamble_overlap"]["train:test"] == 0
    with pytest.raises(ValueError):
        load_split(tmp_path, "test.0")


def test_fit_inputs():
    class Tokenizer:
        def fit_on_texts(self, texts):
            self.texts = texts

    tokenizer = Tokenizer()
    train = pd.DataFrame({"text": ["train only"], "labels": [0]})
    assert fit_training_texts(train, tokenizer, lambda rows: rows) == [["train", "only"]]
    assert tokenizer.texts == ["train only"]
