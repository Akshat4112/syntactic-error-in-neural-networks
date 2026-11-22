import json
import re
from pathlib import Path

import pandas as pd

from .validation import parse_record, SPLITS

CLEAN_REGEX = re.compile(r"[.(),``!?:;\-='...@_]")
PRIMARY_SPLITS = ("train", "validation", "test")


def clean_text(s):
    return ' '.join(CLEAN_REGEX.sub(" ", s).split())


def load_split(project_root, name):
    if name not in PRIMARY_SPLITS:
        raise ValueError(f"Unknown primary split: {name}")
    path = Path(project_root) / "data" / "rnn_agr_simple" / SPLITS[name]
    with path.open(encoding="utf-8") as stream:
        records = [parse_record(line) for line in stream]
    if not records:
        raise ValueError(f"Empty split: {name}")
    return pd.DataFrame(records, columns=["labels", "text"])


def fit_training_texts(train, tokenizer, embedding_factory):
    texts = train["text"].tolist()
    tokenizer.fit_on_texts(texts)
    return embedding_factory([text.split() for text in texts])


class dataset:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root or Path.cwd()).resolve()

    def download_data(self, sha256=None, offline=False):
        from .fetch import fetch_dataset
        return fetch_dataset(self.project_root, sha256, offline)

    def preprocess_data(self):
        data_dir = self.project_root / "data"
        frames = {name: load_split(self.project_root, name)
                  for name in PRIMARY_SPLITS}
        unique = {name: set(frame["text"]) for name, frame in frames.items()}
        overlap = {}
        for index, left in enumerate(PRIMARY_SPLITS):
            for right in PRIMARY_SPLITS[index + 1:]:
                overlap[f"{left}:{right}"] = len(unique[left] & unique[right])
        for name, frame in frames.items():
            frame.to_csv(data_dir / f"{name}_df.csv", index=False)
        report = {"unique_exact_preamble_overlap": overlap,
                  "counts": {name: len(frame) for name, frame in frames.items()}}
        (data_dir / "split-overlap.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Data Preprocessed and Saved...")
