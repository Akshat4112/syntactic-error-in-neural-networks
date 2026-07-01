import argparse

from dataset import dataset
from train import training_model
from evaluation import evaluation


def main():
    parser = argparse.ArgumentParser(description="Syntactic Agreement in Neural Networks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("preprocess", help="Download and preprocess data")
    prep.add_argument("--download", action="store_true", help="Download raw data first")

    train = subparsers.add_parser("train", help="Train a model")
    train.add_argument("--model", choices=["lstm", "bert"], required=True, help="Model to train")
    train.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")

    evl = subparsers.add_parser("evaluate", help="Evaluate a trained model against human data")
    evl.add_argument("--model", choices=["lstm", "rnn", "bert"], required=True, help="Model to evaluate")

    args = parser.parse_args()

    if args.command == "preprocess":
        data_obj = dataset()
        if args.download:
            data_obj.download_data()
        data_obj.preprocess_data()

    elif args.command == "train":
        train_obj = training_model()
        train_obj.prepare_training()
        if args.model == "lstm":
            train_obj.train_lstm()
        elif args.model == "bert":
            train_obj.train_bert_hugging_face(num_epochs=args.epochs)

    elif args.command == "evaluate":
        eval_obj = evaluation()
        if args.model == "lstm":
            eval_obj.evaluate_lstm()
        elif args.model == "rnn":
            eval_obj.evaluate_rnn()
        elif args.model == "bert":
            eval_obj.evaluate_bert()


if __name__ == "__main__":
    main()
