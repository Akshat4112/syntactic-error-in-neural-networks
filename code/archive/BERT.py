from simpletransformers.classification import ClassificationModel, ClassificationArgs
import sklearn.metrics
import torch
import logging

def train_bert(self):
    """
    Trains a BERT model on the provided training data.

    Returns:
        None
    """
    # logging.basicConfig(level=logging.INFO)
    # transformers_logger = logging.getLogger("transformers")
    # transformers_logger.setLevel(logging.WARNING)

    sample_df = self.df_train[:100]
    eval_df = self.df_train[:1000]
    print(sample_df.head())

    model_args = ClassificationArgs(num_train_epochs=10, overwrite_output_dir=True, train_batch_size=16,
                                    evaluate_during_training=True)
    model = ClassificationModel("bert", "bert-base-cased", args=model_args, num_labels=2)

    model.train_model(train_df=sample_df, eval_df=eval_df, show_running_loss=True,
                      acc=sklearn.metrics.accuracy_score)

    test_sample_df = self.df_train.sample(n=100)
    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(test_sample_df)
    print(result)
    precision = result['tp'] / (result['tp'] + result['fp'])
    recall = result['tp'] / (result['tp'] + result['fn'])
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(precision, recall, f1_score)

    model.model.save_pretrained('../models/bert-finetuned')
    model.tokenizer.save_pretrained('../models/bert-finetuned')
    model.config.save_pretrained('../models/bert-finetuned/')