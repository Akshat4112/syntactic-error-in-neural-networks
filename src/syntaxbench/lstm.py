"""Inference for the active LSTM and its fitted tokenizer."""
from pathlib import Path


def load_lstm_predictor(model_dir):
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    from tensorflow.keras.utils import pad_sequences

    model_dir = Path(model_dir)
    tokenizer_path = model_dir / "LSTM_tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError("Missing LSTM tokenizer; retrain the model.")
    tokenizer = tokenizer_from_json(tokenizer_path.read_text(encoding="utf-8"))
    model = load_model(model_dir / "LSTM_model.keras", compile=False)
    max_len = model.input_shape[-1]
    if not isinstance(max_len, int) or max_len <= 0:
        raise ValueError("LSTM must have a fixed positive input length.")

    def predict(text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Expected a nonempty text preamble.")
        sequence = tokenizer.texts_to_sequences([text])
        encoded = pad_sequences(sequence, maxlen=max_len,
                                padding="post", truncating="pre")
        return int(model.predict(encoded, verbose=0)[0].argmax())

    return predict
