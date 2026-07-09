# Syntactic Agreement in Neural Networks

Can neural networks learn grammar? This project investigates whether LSTM and BERT models can learn **subject-verb number agreement** — predicting whether a verb should be singular or plural based on its preceding context (the "preamble").

The task is based on [Linzen, Dupoux & Goldberg (2016)](https://aclanthology.org/Q16-1037/), "Assessing the ability of LSTMs to learn syntax-sensitive dependencies." Model predictions are compared against human behavioral data from psycholinguistic experiments (speeded reading, self-paced reading, and unspeeded judgment tasks).

## Results

| Model | Test Accuracy | Training Epochs |
|-------|--------------|-----------------|
| Majority class | ~68% | — |
| Logistic regression (TF-IDF) | ~92% | — |
| LSTM  | ~93%         | 1               |
| LSTM  | ~99%         | 20              |
| BERT  | ~99%         | 10              |

## Repository Structure

```
code/
  main.py              # CLI entry point (argparse)
  dataset.py           # Data preprocessing (TSV -> CSV)
  train.py             # LSTM (Keras) and BERT (HuggingFace) training
  evaluation.py        # Evaluation against human behavior data
  baselines.py         # Baseline models (majority class, logistic regression)
  analysis.py          # Statistical analysis: human vs model comparison
  reproducibility.py   # Seed control and experiment config saving

data/
  rnn_agr_simple/      # Raw Linzen et al. dataset (~142K examples)
  human_behavior_data/ # Human psycholinguistic experiment data

tests/                 # pytest test suite (41 tests)
results/               # Structured JSON output (gitignored)
figures/               # Training loss/accuracy plots
references/            # Related literature
```

## Setup

**Requirements:** Python >= 3.10

```bash
git clone https://github.com/Akshat4112/Syntactic-Error-in-Neural-Networks.git
cd Syntactic-Error-in-Neural-Networks
pip install -r requirements.txt
```

## Usage

All commands are run from the repository root via the CLI:

```bash
# Preprocess raw data into train/test CSVs
python code/main.py preprocess

# Train models
python code/main.py train --model lstm --epochs 10
python code/main.py train --model bert --epochs 10

# Run baseline models for comparison
python code/main.py baselines

# Evaluate trained models against human behavior data
python code/main.py evaluate --model lstm
python code/main.py evaluate --model bert

# Evaluate accuracy by number of agreement attractors (0-5)
python code/main.py attractor-eval --model lstm
python code/main.py attractor-eval --model bert

# Run statistical analysis comparing models to human behavior
python code/main.py analyze
python code/main.py analyze --models LSTM BERT
```

Run `python code/main.py --help` for full options.

## Running Tests

```bash
pytest              # run all 41 tests
pytest tests/ -v    # verbose output
```

## Methodology

### Task

The task is **binary classification**: given a sentence preamble (everything before the verb), predict whether the following verb should be **singular** (VBZ, label 0) or **plural** (VBP, label 1). The dataset contains ~142K training examples from the Linzen et al. corpus.

### Agreement Attractors

The critical challenge is **agreement attraction** — intervening nouns of the opposite number between the subject and verb. For example, in "the key to the **cabinets**" the verb should be singular (agreeing with "key"), but the plural noun "cabinets" can cause errors in both humans and models. Test splits `numpred.test.{0-5}` group examples by attractor count.

### Models

**Baselines:**
- **Majority class** — always predicts the most frequent label (~68% accuracy)
- **Logistic regression** — TF-IDF features (10K) with 5-fold cross-validation (~92% accuracy)

**Neural models:**
- **LSTM** — Word2Vec embeddings (250d) → LSTM (128 hidden) → Dense layers with dropout → softmax classification. Trained with RMSprop and categorical cross-entropy.
- **BERT** — Fine-tunes `bert-base-cased` for sequence classification using the HuggingFace Trainer API.

### Evaluation

Trained models are evaluated on two axes:

1. **Accuracy by attractor count** — How does performance degrade as more agreement attractors are added? This reveals whether models truly learn hierarchical agreement or rely on linear heuristics.

2. **Comparison with human behavior** — Model predictions are compared against human behavioral data from three psycholinguistic paradigms:
   - **SPEEDED_RSVP** — Rapid serial visual presentation
   - **SPEEDED_SPR** — Self-paced reading
   - **UNSPEEDED** — Untimed grammaticality judgments

### Statistical Analysis

The `analyze` command produces:
- Per-condition accuracy with **95% bootstrap confidence intervals** for both humans and models
- **McNemar's test** for statistical significance of human vs model accuracy differences
- **Pearson correlation** between human and model error patterns across experimental conditions
- **Pairwise model comparisons** (LSTM vs BERT) via McNemar's test
- Accuracy breakdown by **match** (subject-attractor agree) vs **mismatch** (attraction condition)

All results are saved as structured JSON to `results/`.

### Reproducibility

All random seeds are unified at 42 across Python, NumPy, TensorFlow, and PyTorch via `set_seed()`. Training hyperparameters and metrics are automatically saved as JSON configs to `results/`.

## Tech Stack

- TensorFlow / Keras — LSTM training
- PyTorch / HuggingFace Transformers — BERT fine-tuning
- Gensim — Word2Vec embeddings
- scikit-learn — Train/test splitting, baselines
- scipy — Statistical tests
- matplotlib — Visualization

## Citation

This work builds on:

```bibtex
@article{linzen2016assessing,
    Author = {Linzen, Tal and Dupoux, Emmanuel and Goldberg, Yoav},
    Journal = {Transactions of the Association for Computational Linguistics},
    Title = {Assessing the ability of {LSTMs} to learn syntax-sensitive dependencies},
    Volume = {4},
    Pages = {521--535},
    Year = {2016}
}
```

## License

This project is licensed under the MIT License.
