import os
import warnings
from pathlib import Path

import pandas as pd
import tensorflow as tf
import torch
from keras.models import load_model
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification

tf.random.set_seed(7)
warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class evaluation:
    def __init__(self):
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

    def _run_inference(self, model_col, predict_fn):
        for df in self.dataframes:
            predictions = []
            for i in tqdm(range(len(df['Preamble']))):
                predictions.append(predict_fn(df['Preamble'][i]))
            df[model_col] = predictions
        self._save_results()

    def evaluate_lstm(self):
        model = load_model(str(PROJECT_ROOT / 'models' / 'LSTM_model.keras'))
        print("Model loaded successfully")
        self._run_inference('LSTM', lambda text: model.predict(text))

    def evaluate_rnn(self):
        model = load_model(str(PROJECT_ROOT / 'models' / 'model_LSTM_2_epochs.h5'))
        print("Model loaded successfully")
        self._run_inference('RNN', lambda text: model.predict(text))

    def evaluate_bert(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(PROJECT_ROOT / 'models' / 'training_model_bert_full_data')
        )
        print("Model loaded successfully")
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=device)
        self._run_inference('BERT', lambda text: classifier(text)[0]['label'])
