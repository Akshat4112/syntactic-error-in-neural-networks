# CLAUDE.md

## Project Overview

Research project investigating syntactic agreement errors in neural networks. The core question: can neural models (LSTM, BERT) learn subject-verb number agreement (singular vs. plural), and how does their performance compare to human behavior?

Based on the dataset and methodology from Linzen, Dupoux & Goldberg (2016), "Assessing the ability of LSTMs to learn syntax-sensitive dependencies."

## Repository Structure

```
code/                    # Main source code
  main.py                # Entry point — orchestrates data prep, training, evaluation
  dataset.py             # Data download and preprocessing (TSV → CSV)
  train.py               # Model training: LSTM (Keras/TF), BERT (HuggingFace Transformers)
  evaluation.py          # Evaluation against human behavior data
  transformer.py         # Standalone BERT inference script
  archive/               # Earlier/alternative implementations
    BERT.py              # BERT via simpletransformers
    transformer.py       # BERT inference (slightly different paths)

data/
  rnn_agr_simple/        # Raw Linzen et al. dataset (TSV, ~142K training examples)
  train_df.csv           # Preprocessed training data (labels: 0=singular, 1=plural)
  test_df.csv            # Preprocessed test/validation data
  human_behavior_data/   # Human psycholinguistic experiment results (SPEEDED_RSVP, SPEEDED_SPR, UNSPEEDED)

experiment_tracking/
  mlflow.db              # MLflow experiment tracking database (SQLite)

figures/                 # Training plots (loss, accuracy curves)
models/                  # Saved trained models (gitignored)
references/              # Related literature (PDF)
```

## Tech Stack

- **Python 3.11**
- **TensorFlow/Keras** — LSTM model architecture and training
- **PyTorch + HuggingFace Transformers** — BERT fine-tuning and inference
- **Word2Vec (gensim)** — Embedding initialization for LSTM
- **NLTK** — Tokenization (punkt tokenizer)
- **MLflow** — Experiment tracking (SQLite backend)
- **scikit-learn** — Train/test splitting, metrics
- **pandas** — Data manipulation
- **matplotlib** — Training visualization

## How It Works

### Data Pipeline
1. Raw data is tab-separated: POS tag (VBZ=singular, VBP=plural) + sentence preamble
2. `dataset.preprocess_data()` cleans text (removes punctuation), converts POS tags to binary labels (0/1), saves as CSV

### Training Pipeline
1. `training_model.prepare_training()` — Tokenizes text, trains Word2Vec embeddings, builds embedding matrix, prepares train/test split (80/20)
2. Model training options (selected by uncommenting in `main.py`):
   - `train_lstm()` — Sequential LSTM: Embedding(250d) → LSTM(128) → Dense(64) → Dropout → Dense(64) → Dropout → Dense(2, sigmoid). Uses RMSprop, binary crossentropy.
   - `train_bert_hugging_face()` — Fine-tunes `bert-base-cased` for sequence classification via HuggingFace Trainer API.

### Evaluation Pipeline
`evaluation.py` loads trained models and runs inference on human behavior datasets (psycholinguistic experiments), saving model predictions alongside human responses.

### Key Parameters
- Max sequence length: 47 tokens
- Word2Vec embedding dimension: 250
- LSTM hidden size: 128
- Dropout rates: 0.50 (first), 0.20 (second)
- BERT base model: `bert-base-cased`
- Random seed: `tf.random.set_seed(7)`, `random_state=42` for splits

## Running the Code

```bash
pip install -r requirements.txt
cd code
python main.py
```

All scripts use relative paths (`../data/`, `../models/`, etc.) and must be run from the `code/` directory.

## Important Conventions

- **Working directory**: All Python scripts assume CWD is `code/`. Relative paths like `../data/` and `../models/` break otherwise.
- **Model selection**: Training and evaluation methods are toggled by commenting/uncommenting calls in `main.py`. There is no CLI argument parser.
- **GPU expected**: Both `train.py` and `evaluation.py` call `torch.cuda.get_device_name()` on init, which will error without a CUDA GPU.
- **WandB disabled**: `os.environ["WANDB_DISABLED"] = "true"` is set in training and evaluation modules.
- **No test suite**: There are no automated tests. Validation is done via training metrics and manual evaluation against human data.
- **No CI/CD**: No GitHub Actions or other CI configuration exists.
- **No linter config**: No `.flake8`, `pyproject.toml`, or similar; no enforced code style.

## Data Labels

- `0` = singular (originally POS tag `VBZ`)
- `1` = plural (originally POS tag `VBP`)
- BERT classifier output: `LABEL_0` = singular, `LABEL_1` = plural

## Gotchas

- `train.py:tokenize_function` and `compute_metrics` reference module-level `tokenizer` and `metric` variables that are only defined inside `train_bert_hugging_face()`, causing `NameError` if called as standalone methods.
- `code/transformer.py` and `code/archive/transformer.py` are nearly identical files with minor path differences.
- The notebook `data/psycholinguistics-syntactic-understanding-of-nn.ipynb` contains the original exploratory analysis and is not maintained in sync with the modular Python code.
- `.idea/` directory (PyCharm config) is committed to the repo.
