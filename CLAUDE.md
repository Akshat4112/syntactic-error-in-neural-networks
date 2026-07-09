# CLAUDE.md

## Project Overview

Research project investigating syntactic agreement errors in neural networks. The core question: can neural models (LSTM, BERT) learn subject-verb number agreement (singular vs. plural), and how does their performance compare to human behavior?

Based on the dataset and methodology from Linzen, Dupoux & Goldberg (2016), "Assessing the ability of LSTMs to learn syntax-sensitive dependencies."

## Repository Structure

```
code/                    # Main source code
  main.py                # CLI entry point (argparse) — preprocess, train, evaluate, baselines, analyze
  dataset.py             # Data download and preprocessing (TSV → CSV)
  train.py               # Model training: LSTM (Keras/TF), BERT (HuggingFace Transformers)
  evaluation.py          # Evaluation against human behavior data
  baselines.py           # Baseline models (majority class, logistic regression)
  analysis.py            # Statistical analysis: human vs model comparison
  reproducibility.py     # Seed control and experiment config saving
  transformer.py         # Standalone BERT inference script
  archive/
    BERT.py              # Earlier BERT implementation via simpletransformers

data/
  rnn_agr_simple/        # Raw Linzen et al. dataset (TSV, ~142K training examples)
  train_df.csv           # Preprocessed training data (labels: 0=singular, 1=plural)
  test_df.csv            # Preprocessed test/validation data
  human_behavior_data/   # Human psycholinguistic experiment results (SPEEDED_RSVP, SPEEDED_SPR, UNSPEEDED)

tests/                   # pytest test suite (41 tests)
  test_dataset.py        # Tests for text cleaning and POS label conversion
  test_train.py          # Tests for compute_metrics logic
  test_analysis.py       # Tests for bootstrap CI and McNemar's test
  test_baselines.py      # Tests for baseline model logic

results/                 # Structured JSON output from experiments (gitignored)
figures/                 # Training plots (loss, accuracy curves)
models/                  # Saved trained models (gitignored)
references/              # Related literature (PDF)
```

## Tech Stack

- **Python >= 3.10** (developed on 3.11)
- **TensorFlow/Keras** — LSTM model architecture and training
- **PyTorch + HuggingFace Transformers** — BERT fine-tuning and inference
- **Word2Vec (gensim)** — Embedding initialization for LSTM
- **scikit-learn** — Train/test splitting, metrics, baselines (TF-IDF + LogisticRegression)
- **scipy** — Statistical tests (McNemar's test, Pearson correlation)
- **pandas** — Data manipulation
- **matplotlib** — Training visualization
- **pytest** — Test suite
- **flake8** — Linting (config in `.flake8`, max line length 120)

## CLI Usage

```bash
pip install -r requirements.txt

# Preprocess raw data into train/test CSVs
python code/main.py preprocess
python code/main.py preprocess --download   # download raw data first

# Train a model
python code/main.py train --model lstm
python code/main.py train --model bert --epochs 20

# Evaluate against human behavior data
python code/main.py evaluate --model lstm
python code/main.py evaluate --model bert
python code/main.py evaluate --model rnn

# Evaluate accuracy by number of agreement attractors (0-5)
python code/main.py attractor-eval --model lstm
python code/main.py attractor-eval --model bert

# Run baseline models (majority class, logistic regression)
python code/main.py baselines

# Run statistical analysis comparing models to human behavior
python code/main.py analyze
python code/main.py analyze --models LSTM BERT
```

## Running Tests

```bash
pytest                  # runs all 41 tests
pytest tests/ -v        # verbose output
```

Tests use `pythonpath = ["code"]` (configured in `pyproject.toml`) so they can import from `code/` directly.

## How It Works

### Data Pipeline
1. Raw data is tab-separated: POS tag (VBZ=singular, VBP=plural) + sentence preamble
2. `Dataset.preprocess_data()` cleans text via regex (removes punctuation), converts POS tags to binary labels (0/1), saves as CSV

### Training Pipeline
1. `TrainingModel.prepare_training()` — Tokenizes text, trains Word2Vec embeddings, builds embedding matrix, prepares train/test split (80/20)
2. Model training:
   - **LSTM**: Embedding(250d) → LSTM(128) → Dense(64) → Dropout(0.5) → Dense(64) → Dropout(0.2) → Dense(2, softmax). RMSprop optimizer, categorical crossentropy.
   - **BERT**: Fine-tunes `bert-base-cased` for sequence classification via HuggingFace Trainer.
3. Hyperparameters and final metrics are saved to `results/` as JSON via `save_config()`.

### Evaluation Pipeline
Loads trained models, runs batch inference on human behavior datasets (psycholinguistic experiments), saves predictions alongside human responses. The `attractor-eval` command measures accuracy on test splits grouped by number of agreement attractors (0-5) to analyze how intervening nouns affect model performance.

### Statistical Analysis Pipeline
The `analyze` command compares model predictions to human behavioral data across three psycholinguistic paradigms:
- **Per-condition accuracy** with 95% bootstrap confidence intervals for both human and model
- **McNemar's test** for statistical significance of human vs model accuracy differences
- **Pearson correlation** between human and model error patterns across conditions
- **Pairwise model comparisons** (LSTM vs BERT) via McNemar's test
- **Match/mismatch breakdown** — accuracy when subject-attractor number agrees vs disagrees
- All results saved as structured JSON to `results/human_model_comparison.json`

### Baselines
- **Majority class**: Always predicts the most frequent label (~68% accuracy)
- **Logistic regression**: TF-IDF features (10K) + LogisticRegression with 5-fold cross-validation (~92% accuracy)

### Human Behavior Data Conditions
The experimental conditions encode subject-attractor agreement patterns:
- `ss_*` / `pp_*` — subject and attractor match in number (easier)
- `sp_*` / `ps_*` — subject and attractor mismatch (agreement attraction, harder)
- `*_prep` — prepositional phrase modifier
- `*_rel` — relative clause modifier

### Key Parameters
- Max sequence length: 47 tokens
- Word2Vec embedding dimension: 250
- LSTM hidden size: 128
- Dropout rates: 0.50 (first), 0.20 (second)
- BERT base model: `bert-base-cased`
- Random seed: 42 (unified across all frameworks via `set_seed()`)

### Reproducibility
All random seeds are unified through `reproducibility.set_seed(42)`, which sets seeds for Python's `random`, NumPy, TensorFlow, and PyTorch (including CUDA deterministic mode). Training configs and metrics are automatically saved as JSON to `results/`.

## Data Labels

- `0` = singular (POS tag `VBZ`)
- `1` = plural (POS tag `VBP`)
- BERT classifier output: `LABEL_0` = singular, `LABEL_1` = plural

## Path Resolution

All source files use `pathlib.Path(__file__).resolve().parent.parent` as `PROJECT_ROOT` to locate data, models, and figures. Scripts can be run from any working directory.

## CI

GitHub Actions runs on push/PR to `main`:
1. Lint with flake8
2. Run pytest (41 tests)

Config: `.github/workflows/ci.yml`
