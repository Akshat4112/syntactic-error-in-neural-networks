import argparse
from pathlib import Path


def main(argv=None, project_root=None):
    parser = argparse.ArgumentParser(description="Syntactic Agreement in Neural Networks")
    parser.add_argument("--project-root", type=Path, default=project_root or Path.cwd(),
                        help="Workspace containing data, models and figures (default: current directory)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="Explicitly retrieve verified raw data")
    fetch.add_argument("--sha256", required=True, help="Trusted archive SHA-256")
    fetch.add_argument("--offline", action="store_true")

    prep = subparsers.add_parser("preprocess", help="Preprocess data")
    prep.add_argument("--download", action="store_true", help="Download raw data first")

    prep.add_argument("--sha256", help="Trusted archive SHA-256 for explicit download")

    train = subparsers.add_parser("train", help="Train a model")
    train.add_argument("--model", choices=["lstm", "bert"], required=True, help="Model to train")
    train.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")

    evl = subparsers.add_parser("evaluate", help="Evaluate a trained model against human data")
    evl.add_argument("--model", choices=["lstm", "rnn", "bert"], required=True, help="Model to evaluate")

    args = parser.parse_args(argv)

    if args.command == "fetch":
        from .fetch import fetch_dataset
        try:
            print(fetch_dataset(args.project_root, args.sha256, args.offline))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))

    elif args.command == "preprocess":
        from .dataset import dataset
        data_obj = dataset(project_root=args.project_root)
        if args.download:
            if not args.sha256:
                parser.error("--download requires --sha256")
            data_obj.download_data(args.sha256)
        data_obj.preprocess_data()

    elif args.command == "train":
        from .train import training_model
        train_obj = training_model(project_root=args.project_root)
        train_obj.prepare_training()
        if args.model == "lstm":
            train_obj.train_lstm(num_epochs=args.epochs)
        elif args.model == "bert":
            train_obj.train_bert_hugging_face(num_epochs=args.epochs)

    elif args.command == "evaluate":
        from .evaluation import evaluation
        eval_obj = evaluation(project_root=args.project_root)
        if args.model == "lstm":
            eval_obj.evaluate_lstm()
        elif args.model == "rnn":
            eval_obj.evaluate_rnn()
        elif args.model == "bert":
            eval_obj.evaluate_bert()


if __name__ == "__main__":
    main()
