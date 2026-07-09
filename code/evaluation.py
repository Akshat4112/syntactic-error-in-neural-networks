import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Evaluation:
    def __init__(self):
        import tensorflow as tf
        import torch
        tf.random.set_seed(42)
        print("Num GPUs Available for tf: ", len(tf.config.list_physical_devices('GPU')))
        print(f'PyTorch version: {torch.__version__}')
        if torch.cuda.is_available():
            print(f'CUDNN version: {torch.backends.cudnn.version()}')
            print(f'Available GPU devices for Torch: {torch.cuda.device_count()}')
            print(f'Device Name: {torch.cuda.get_device_name()}')
        else:
            print('No CUDA GPU available for PyTorch, using CPU')

        human_data_dir = PROJECT_ROOT / 'data' / 'human_behavior_data'
        self.df_SPEEDED_RSVP = pd.read_csv(human_data_dir / 'SPEEDED_RSVP.csv')
        self.df_SPEEDED_SPR = pd.read_csv(human_data_dir / 'SPEEDED_SPR.csv')
        self.df_UNSPEEDED = pd.read_csv(human_data_dir / 'UNSPEEDED.csv')
        self.dataframes = [self.df_SPEEDED_RSVP, self.df_SPEEDED_SPR, self.df_UNSPEEDED]

    def _save_results(self):
        output_dir = PROJECT_ROOT / 'data' / 'human_behavior_data'
        self.df_SPEEDED_RSVP.to_csv(output_dir / 'df_SPEEDED_RSVP.csv', index=False)
        self.df_SPEEDED_SPR.to_csv(output_dir / 'df_SPEEDED_SPR.csv', index=False)
        self.df_UNSPEEDED.to_csv(output_dir / 'df_UNSPEEDED.csv', index=False)

    def _run_inference_batch(self, model_col, predict_fn):
        for df in self.dataframes:
            preambles = df['Preamble'].tolist()
            predictions = predict_fn(preambles)
            df[model_col] = predictions
        self._save_results()

    def _build_keras_tokenizer(self):
        try:
            from keras.preprocessing.text import Tokenizer
        except ImportError:
            from tensorflow.keras.preprocessing.text import Tokenizer
        train_df = pd.read_csv(PROJECT_ROOT / 'data' / 'train_df.csv')
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_df['text'])
        return tokenizer

    def _keras_predict_batch(self, model, tokenizer, texts, batch_size=128):
        try:
            from keras.utils import pad_sequences
        except ImportError:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
        all_preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            sequences = tokenizer.texts_to_sequences(batch)
            padded = pad_sequences(sequences, maxlen=47, padding='post')
            preds = model.predict(padded, batch_size=batch_size, verbose=0)
            all_preds.extend(np.argmax(preds, axis=-1).tolist())
        return all_preds

    def evaluate_lstm(self):
        from keras.models import load_model
        model = load_model(str(PROJECT_ROOT / 'models' / 'LSTM_model.keras'))
        print("Model loaded successfully")
        tokenizer = self._build_keras_tokenizer()
        self._run_inference_batch(
            'LSTM',
            lambda texts: self._keras_predict_batch(model, tokenizer, texts)
        )

    def evaluate_rnn(self):
        from keras.models import load_model
        model = load_model(str(PROJECT_ROOT / 'models' / 'model_LSTM_2_epochs.h5'))
        print("Model loaded successfully")
        tokenizer = self._build_keras_tokenizer()
        self._run_inference_batch(
            'RNN',
            lambda texts: self._keras_predict_batch(model, tokenizer, texts)
        )

    def evaluate_bert(self):
        import torch
        from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(PROJECT_ROOT / 'models' / 'training_model_bert_full_data')
        )
        print("Model loaded successfully")
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline(
            task="text-classification", model=model, tokenizer=tokenizer,
            device=device, batch_size=32
        )

        def predict_batch(texts):
            results = classifier(texts, batch_size=32)
            return [r['label'] for r in results]

        self._run_inference_batch('BERT', predict_batch)

    def evaluate_by_attractor_count(self, model_name, predict_fn):
        from dataset import clean_text
        raw_dir = PROJECT_ROOT / 'data' / 'rnn_agr_simple'
        results = {}
        for n in range(6):
            test_file = raw_dir / f'numpred.test.{n}'
            if not test_file.exists():
                continue
            df = pd.read_csv(test_file, sep='\t', names=['POS', 'Preamble'])
            df['Preamble'] = df['Preamble'].apply(clean_text)
            labels = df['POS'].map({'VBZ': 0, 'VBP': 1}).values
            predictions = predict_fn(df['Preamble'].tolist())
            pred_labels = []
            for p in predictions:
                if isinstance(p, str):
                    pred_labels.append(int(p.replace('LABEL_', '')))
                else:
                    pred_labels.append(int(p))
            accuracy = (np.array(pred_labels) == labels).mean()
            results[n] = {'accuracy': accuracy, 'count': len(df)}
            print(f"  {n} attractors: {accuracy:.4f} ({len(df)} examples)")
        print(f"\n{model_name} accuracy by attractor count:")
        for n, r in sorted(results.items()):
            print(f"  {n}: {r['accuracy']:.4f} ({r['count']} examples)")
        return results
