# LSTM agreement classifier

The active training path retains the historical 250-dimensional Word2Vec
initialization, 47-token limit, 128-unit LSTM, two 64-unit dense layers,
dropout rates 0.50 and 0.20, RMSprop and categorical cross-entropy.
The official validation partition is used for validation; the tokenizer and
Word2Vec are fitted on training text only.

Changes from the historical configuration: unknown words now map to an
explicit OOV token, zero padding is masked, and truncation retains the last
47 tokens. The fitted tokenizer is saved alongside the Keras model.
Existing checkpoints without the tokenizer must be retrained rather than
silently guessing the vocabulary. Historical scores are not directly
comparable without a controlled rerun.

After training using the active training_model API, obtain canonical
predictions (0 singular, 1 plural) using:

```python
from syntaxbench.lstm import load_lstm_predictor

predict = load_lstm_predictor("models")
label = predict("the keys to the cabinet")
```

Use only trusted locally generated model/tokenizer pairs. Checksummed
artifact manifests and compatibility validation are tracked separately in
SYN-015. The CPU smoke test exercises actual training and reloaded text
inference; it does not establish held-out accuracy.

```sh
python -m pytest -v tests/test_ml_split_runtime.py
```
