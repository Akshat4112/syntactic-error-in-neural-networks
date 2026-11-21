"""Compatibility import for the historical repository layout."""
from pathlib import Path

from syntaxbench.train import training_model as _training_model


class training_model(_training_model):
    def __init__(self, project_root=None):
        super().__init__(project_root or Path(__file__).resolve().parent.parent)
