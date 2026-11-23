"""Optional ML runtime regression; run with the pinned ML extra."""
import numpy as np
import pytest

from syntaxbench.dataset import dataset


def test_real_training_paths(tmp_path, monkeypatch):
    pytest.importorskip("tensorflow")
    pytest.importorskip("transformers")
    from syntaxbench import train
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    raw = tmp_path / "data" / "rnn_agr_simple"
    raw.mkdir(parents=True)
    (raw / "numpred.train").write_text("VBZ\tthe dog\nVBP\tthe dogs\n")
    (raw / "numpred.val").write_text("VBP\tvalidation dogs\n")
    (raw / "numpred.test").write_text("VBZ\ttest dog\n")
    dataset(tmp_path).preprocess_data()
    model = train.training_model(tmp_path)
    model.prepare_training()
    assert model.x_train.shape[0] == 2
    assert model.x_val.shape[0] == 1
    assert model.y_val.tolist() == [[0, 1]]
    assert model.vocab_size == 5
    assert 'validation' not in model.tokenizer.word_index
    assert 'test' not in model.tokenizer.word_index
    assert model.x_val[0, 0] == model.tokenizer.word_index['<OOV>']
    assert np.all(model.x_val[0, 2:] == 0)
    model.train_lstm(num_epochs=1)
    restored = train.tf.keras.models.load_model(tmp_path / "models" / "LSTM_model.keras")
    probs = restored.predict(model.x_val, verbose=0)
    assert probs.shape == (1, 2)
    assert restored.layers[0].mask_zero
    from syntaxbench.lstm import load_lstm_predictor
    predict = load_lstm_predictor(tmp_path / 'models')
    assert predict('validation dogs') == int(probs[0].argmax())
    assert predict('unseen word') in (0, 1)
    with pytest.raises(ValueError):
        predict('  ')
    (tmp_path / 'models' / 'LSTM_tokenizer.json').unlink()
    with pytest.raises(FileNotFoundError):
        load_lstm_predictor(tmp_path / 'models')

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nthe\ndog\ndogs\nvalidation\ntest\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), model_max_length=8)
    config = BertConfig(vocab_size=len(tokenizer), hidden_size=16,
                        num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, num_labels=2)
    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(train.AutoModelForSequenceClassification, "from_pretrained",
                        lambda *a, **k: BertForSequenceClassification(config))

    class Accuracy:
        def compute(self, predictions, references):
            return {"accuracy": float(np.mean(predictions == references))}

    monkeypatch.setattr(train.evaluate, "load", lambda *a: Accuracy())
    original = train.Trainer

    class CheckedTrainer(original):
        def __init__(self, *args, **kwargs):
            assert kwargs["train_dataset"]["text"] == ["the dog", "the dogs"]
            assert kwargs["eval_dataset"]["text"] == ["validation dogs"]
            super().__init__(*args, **kwargs)

        def predict(self, test_dataset, *args, **kwargs):
            assert test_dataset["text"] == ["test dog"]
            result = super().predict(test_dataset, *args, **kwargs)
            assert result.predictions.shape == (1, 2)
            return result

    monkeypatch.setattr(train, "Trainer", CheckedTrainer)
    model.train_bert_hugging_face(num_epochs=1)
    assert (tmp_path / "models" / "training_model_bert_full_data" / "config.json").exists()
