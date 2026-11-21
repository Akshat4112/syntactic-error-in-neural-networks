"""Compatibility import for the historical repository layout."""
from pathlib import Path

from syntaxbench.dataset import clean_text
from syntaxbench.dataset import dataset as _dataset


__all__ = ["clean_text", "dataset"]


class dataset(_dataset):
    def __init__(self, project_root=None):
        super().__init__(project_root or Path(__file__).resolve().parent.parent)
