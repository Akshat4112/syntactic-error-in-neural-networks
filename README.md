# Syntactic Agreement in Neural Networks

Can neural networks learn grammar? This project investigates whether LSTM and BERT models can learn **subject-verb number agreement** — predicting whether a verb should be singular or plural based on its preceding context (the "preamble").

The task is based on [Linzen, Dupoux & Goldberg (2016)](https://aclanthology.org/Q16-1037/), "Assessing the ability of LSTMs to learn syntax-sensitive dependencies." Model predictions are compared against human behavioral data from psycholinguistic experiments (speeded reading, self-paced reading, and unspeeded judgment tasks).

## Results

| Model | Validation Accuracy | Training Epochs |
|-------|-------------------|-----------------|
| LSTM  | ~93%              | 1               |
| LSTM  | ~99%              | 20              |
| BERT  | ~99%              | 10              |

## Repository Structure

```
code/
  main.py              # CLI entry point (argparse)
  dataset.py           # Data preprocessing (TSV → CSV)
  train.py             # LSTM (Keras) and BERT (HuggingFace) training
  evaluation.py        # Evaluation against human behavior data
  transformer.py       # Standalone BERT inference script

data/
  rnn_agr_simple/      # Raw Linzen et al. dataset (~142K examples)
  train_df.csv         # Preprocessed training data
  test_df.csv          # Preprocessed test data
  human_behavior_data/ # Human psycholinguistic experiment data

tests/                 # pytest test suite (26 tests)
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

# Evaluate trained models against human behavior data
python code/main.py evaluate --model lstm
python code/main.py evaluate --model bert

# Evaluate accuracy by number of agreement attractors (0-5)
python code/main.py attractor-eval --model lstm
python code/main.py attractor-eval --model bert
```

Run `python code/main.py --help` for full options.

## Running Tests

```bash
pytest              # run all 26 tests
pytest tests/ -v    # verbose output
```

## How It Works

### Data

The dataset contains ~142K sentence preambles labeled with the verb form that should follow:
- **VBZ** (singular) → label `0` — e.g., "the cat" → *runs*
- **VBP** (plural) → label `1` — e.g., "the cats" → *run*

The challenge increases with **agreement attractors** — nouns of the opposite number between the subject and verb (e.g., "the key to the **cabinets**" should still take a singular verb).

### Models

**LSTM:** Word2Vec embeddings (250d) → LSTM (128 hidden) → Dense layers with dropout → softmax classification. Trained with RMSprop and categorical cross-entropy.

**BERT:** Fine-tunes `bert-base-cased` for sequence classification using the HuggingFace Trainer API.

### Evaluation

Trained models are evaluated against human behavioral data from three psycholinguistic paradigms (SPEEDED_RSVP, SPEEDED_SPR, UNSPEEDED) to compare neural network agreement patterns with human reading behavior.

## Tech Stack

- TensorFlow / Keras — LSTM training
- PyTorch / HuggingFace Transformers — BERT fine-tuning
- Gensim — Word2Vec embeddings
- scikit-learn — Train/test splitting
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
