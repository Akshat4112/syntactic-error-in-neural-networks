"""Compatibility entry point; install the project before use."""
from pathlib import Path

from syntaxbench.cli import main

if __name__ == "__main__":
    main(project_root=Path(__file__).resolve().parent.parent)
