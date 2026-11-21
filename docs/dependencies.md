# Dependency profiles

Use Python 3.10 or 3.11 in a fresh virtual environment. CI currently validates
3.11 on Linux. Other platforms are not yet certified. The project metadata
installs dependencies only until the package migration in SYN-004; invoke the
existing CLI from the repository root.

| Profile | Command | Purpose |
| --- | --- | --- |
| Core | `python -m pip install .` | Dataset preprocessing |
| Development | `python -m pip install '.[dev]'` | Core plus lint and tests |
| ML | `python -m pip install '.[ml,dev]'` | Active LSTM/BERT pipeline and tests |

For Linux CPU-only PyTorch, install the CPU wheel before the ML profile:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[ml,dev]'
python -m pip check
PYTHONPATH=code python -c "import dataset, train, evaluation, transformer"
python code/main.py --help
python -m pytest
```

Direct dependencies are pinned in pyproject.toml; transitive dependencies
remain resolver-selected, so this is not a complete environment lock.
TensorFlow/Keras 2.15 preserve the legacy Keras API; NumPy 1.26 and SciPy
1.12 retain compatibility with Gensim 4.3.2. Evaluate, PyTorch and Accelerate
are now explicit rather than relying on incidental dependencies.
The archive extra is separate and is not validated by active-pipeline CI.
The original 2023 environment is preserved in requirements/legacy-2023.txt
for historical reference, not recommended as an install target. It contains
platform-specific and outdated pins. No model performance claims are made
by this migration; full training and inference repairs are later tasks.
