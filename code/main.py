import argparse


def main():
    parser = argparse.ArgumentParser(description="Syntactic Agreement in Neural Networks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("preprocess", help="Download and preprocess data")
    prep.add_argument("--download", action="store_true", help="Download raw data first")

    train = subparsers.add_parser("train", help="Train a model")
    train.add_argument("--model", choices=["lstm", "bert"], required=True, help="Model to train")
    train.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    train.add_argument("--max-steps", type=int, default=-1,
                       help="Max training steps (overrides epochs if > 0, useful for CPU training)")

    evl = subparsers.add_parser("evaluate", help="Evaluate a trained model against human data")
    evl.add_argument("--model", choices=["lstm", "rnn", "bert"], required=True, help="Model to evaluate")

    attr = subparsers.add_parser("attractor-eval", help="Evaluate accuracy by attractor count (0-5)")
    attr.add_argument("--model", choices=["lstm", "bert"], required=True, help="Model to evaluate")

    subparsers.add_parser("baselines", help="Run baseline models (majority class, logistic regression)")

    analyze = subparsers.add_parser("analyze", help="Run human vs model comparison analysis")
    analyze.add_argument("--models", nargs="+", default=["LSTM", "BERT"],
                         help="Model columns to analyze (default: LSTM BERT)")

    args = parser.parse_args()

    if args.command == "preprocess":
        from dataset import Dataset
        data_obj = Dataset()
        if args.download:
            data_obj.download_data()
        data_obj.preprocess_data()

    elif args.command == "train":
        from train import TrainingModel
        train_obj = TrainingModel()
        train_obj.prepare_training()
        if args.model == "lstm":
            train_obj.train_lstm(num_epochs=args.epochs)
        elif args.model == "bert":
            train_obj.train_bert_hugging_face(num_epochs=args.epochs, max_steps=args.max_steps)

    elif args.command == "evaluate":
        from evaluation import Evaluation
        eval_obj = Evaluation()
        if args.model == "lstm":
            eval_obj.evaluate_lstm()
        elif args.model == "rnn":
            eval_obj.evaluate_rnn()
        elif args.model == "bert":
            eval_obj.evaluate_bert()

    elif args.command == "attractor-eval":
        from evaluation import Evaluation
        import numpy as np
        eval_obj = Evaluation()
        if args.model == "lstm":
            from keras.models import load_model
            from pathlib import Path
            import pandas as pd
            try:
                from keras.preprocessing.text import Tokenizer
                from keras.utils import pad_sequences
            except ImportError:
                from tensorflow.keras.preprocessing.text import Tokenizer
                from tensorflow.keras.preprocessing.sequence import pad_sequences

            PROJECT_ROOT = Path(__file__).resolve().parent.parent
            model = load_model(str(PROJECT_ROOT / 'models' / 'LSTM_model.keras'))
            train_df = pd.read_csv(PROJECT_ROOT / 'data' / 'train_df.csv')
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(train_df['text'])

            def predict_fn(texts):
                sequences = tokenizer.texts_to_sequences(texts)
                padded = pad_sequences(sequences, maxlen=47, padding='post')
                preds = model.predict(padded, batch_size=256, verbose=0)
                return np.argmax(preds, axis=-1).tolist()

            eval_obj.evaluate_by_attractor_count('LSTM', predict_fn)

        elif args.model == "bert":
            import torch
            from pathlib import Path
            from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
            PROJECT_ROOT = Path(__file__).resolve().parent.parent
            tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            bert_model = AutoModelForSequenceClassification.from_pretrained(
                str(PROJECT_ROOT / 'models' / 'training_model_bert_full_data')
            )
            device = 0 if torch.cuda.is_available() else -1
            classifier = pipeline(
                task="text-classification", model=bert_model, tokenizer=tokenizer,
                device=device, batch_size=64
            )

            def predict_fn(texts):
                results = classifier(texts, batch_size=64)
                return [r['label'] for r in results]

            eval_obj.evaluate_by_attractor_count('BERT', predict_fn)

    elif args.command == "baselines":
        from baselines import Baselines
        Baselines().run_all()

    elif args.command == "analyze":
        from analysis import HumanModelComparison
        HumanModelComparison().run_full_analysis(model_cols=args.models)


if __name__ == "__main__":
    main()
