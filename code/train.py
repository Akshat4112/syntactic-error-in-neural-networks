import os
from pathlib import Path

import datasets
import evaluate
import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from datasets import Dataset as HFDataset
from keras.initializers import Constant
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer, AutoTokenizer

try:
    from keras.preprocessing.text import Tokenizer
    from keras.utils import pad_sequences, to_categorical
    from keras.optimizers import RMSprop
except ImportError:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import RMSprop

from reproducibility import set_seed, save_config

os.environ["WANDB_DISABLED"] = "true"
set_seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TrainingModel:
    def __init__(self):
        print("Num GPUs Available for tf: ", len(tf.config.list_physical_devices('GPU')))
        print(f'PyTorch version: {torch.__version__}')
        if torch.cuda.is_available():
            print(f'CUDNN version: {torch.backends.cudnn.version()}')
            print(f'Available GPU devices for Torch: {torch.cuda.device_count()}')
            print(f'Device Name: {torch.cuda.get_device_name()}')
        else:
            print('No CUDA GPU available for PyTorch, using CPU')

        self.df_train = pd.read_csv(PROJECT_ROOT / 'data' / 'train_df.csv')
        self.df_test = pd.read_csv(PROJECT_ROOT / 'data' / 'test_df.csv')

    def prepare_training(self):
        sentences = [pre.strip().split() for pre in self.df_train['text']]

        w2v_model = gensim.models.Word2Vec(sentences=sentences, vector_size=250, window=10, min_count=1, epochs=10)
        print("Vocab Length is: ", len(w2v_model.wv))

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.df_train['text'])
        encoded_vec = tokenizer.texts_to_sequences(self.df_train['text'])

        self.max_len = 47
        self.vocab_size = len(tokenizer.word_index) + 1
        self.embedding_dim = 250

        pad = pad_sequences(encoded_vec, maxlen=self.max_len, padding='post')

        self.embed_matrix = np.zeros(shape=(self.vocab_size, self.embedding_dim))
        for word, i in tokenizer.word_index.items():
            if word in w2v_model.wv:
                self.embed_matrix[i] = w2v_model.wv[word]

        Y = to_categorical(self.df_train['labels'])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(pad, Y, test_size=0.20, random_state=42)

    def train_lstm(self, num_epochs=2):
        (PROJECT_ROOT / "models").mkdir(exist_ok=True)
        (PROJECT_ROOT / "figures").mkdir(exist_ok=True)
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_len,
                            embeddings_initializer=Constant(self.embed_matrix)))
        model.add(LSTM(128, return_sequences=False))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.50))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.20))
        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
        History = model.fit(self.x_train, self.y_train, epochs=num_epochs, batch_size=64, validation_split=0.2)

        model.save(str(PROJECT_ROOT / "models" / "LSTM_model.keras"))

        save_config('lstm', {
            'embedding_dim': self.embedding_dim,
            'lstm_hidden': 128,
            'dense_units': 64,
            'dropout_1': 0.50,
            'dropout_2': 0.20,
            'optimizer': 'RMSprop',
            'learning_rate': 1e-3,
            'loss': 'categorical_crossentropy',
            'batch_size': 64,
            'epochs': num_epochs,
            'max_len': self.max_len,
            'vocab_size': self.vocab_size,
        }, metrics={
            'final_train_accuracy': float(History.history['accuracy'][-1]),
            'final_val_accuracy': float(History.history['val_accuracy'][-1]),
            'final_train_loss': float(History.history['loss'][-1]),
            'final_val_loss': float(History.history['val_loss'][-1]),
        })

        # Plot training & validation accuracy and loss curves
        plt.figure()
        plt.plot(History.history['loss'])
        plt.plot(History.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(str(PROJECT_ROOT / "figures" / "model_loss_LSTM.png"))

        plt.figure()
        plt.plot(History.history['accuracy'])
        plt.plot(History.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(str(PROJECT_ROOT / "figures" / "model_accuracy_LSTM.png"))
        print("Figures saved successfully")

    def train_bert_hugging_face(self, num_epochs=10):
        ds_train = HFDataset.from_pandas(self.df_train)
        ds_test = HFDataset.from_pandas(self.df_test)
        ds = datasets.DatasetDict({"train": ds_train, "test": ds_test})
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        tokenized_datasets = ds.map(tokenize_function, batched=True)
        split_train_dataset = tokenized_datasets["train"].train_test_split(test_size=0.2)

        train_dataset = split_train_dataset["train"]
        eval_dataset = split_train_dataset["test"]
        test_dataset = tokenized_datasets["test"]
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        training_args = TrainingArguments(
            output_dir=str(PROJECT_ROOT / "models" / "test_trainer"),
            evaluation_strategy="epoch",
            report_to=None,
            num_train_epochs=num_epochs,
            seed=42,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(str(PROJECT_ROOT / "models" / "training_model_bert_full_data"))
        test_results = trainer.predict(test_dataset)
        test_acc = float((np.argmax(test_results.predictions, axis=-1) == test_results.label_ids).mean())
        save_config('bert', {
            'base_model': 'bert-base-cased',
            'num_labels': 2,
            'epochs': num_epochs,
            'evaluation_strategy': 'epoch',
        }, metrics={
            'test_accuracy': test_acc,
            'test_loss': float(test_results.metrics.get('test_loss', 0)),
        })
