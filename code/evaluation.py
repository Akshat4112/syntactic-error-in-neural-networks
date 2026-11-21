"""Compatibility import for the historical repository layout."""
from pathlib import Path

from syntaxbench.evaluation import evaluation as _evaluation


class evaluation(_evaluation):
    def __init__(self, project_root=None):
        super().__init__(project_root or Path(__file__).resolve().parent.parent)
