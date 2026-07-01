import numpy as np


def compute_metrics(eval_pred):
    """Replicate the compute_metrics logic from train.py.

    The original uses an external `metric.compute` call, but the core logic
    is: predictions = argmax(logits, axis=-1), then compare to labels.
    We test the accuracy computation directly.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


class TestComputeMetrics:
    """Test the compute_metrics logic independently of heavy dependencies."""

    def test_perfect_predictions(self):
        logits = np.array([[2.0, 0.1], [0.1, 2.0], [2.0, 0.1]])
        labels = np.array([0, 1, 0])
        result = compute_metrics((logits, labels))
        assert result["accuracy"] == 1.0

    def test_all_wrong_predictions(self):
        logits = np.array([[0.1, 2.0], [2.0, 0.1], [0.1, 2.0]])
        labels = np.array([0, 1, 0])
        result = compute_metrics((logits, labels))
        assert result["accuracy"] == 0.0

    def test_half_correct(self):
        logits = np.array([[2.0, 0.1], [0.1, 2.0], [2.0, 0.1], [0.1, 2.0]])
        labels = np.array([0, 0, 1, 1])
        result = compute_metrics((logits, labels))
        assert result["accuracy"] == 0.5

    def test_single_sample(self):
        logits = np.array([[0.3, 0.7]])
        labels = np.array([1])
        result = compute_metrics((logits, labels))
        assert result["accuracy"] == 1.0

    def test_multiclass_logits(self):
        logits = np.array([
            [0.1, 0.2, 0.9],
            [0.9, 0.1, 0.2],
            [0.1, 0.9, 0.2],
        ])
        labels = np.array([2, 0, 1])
        result = compute_metrics((logits, labels))
        assert result["accuracy"] == 1.0

    def test_argmax_breaks_ties_consistently(self):
        # When logits are equal, np.argmax returns the first index
        logits = np.array([[1.0, 1.0]])
        labels = np.array([0])
        result = compute_metrics((logits, labels))
        assert result["accuracy"] == 1.0

    def test_negative_logits(self):
        logits = np.array([[-1.0, -2.0], [-3.0, -1.0]])
        labels = np.array([0, 1])
        result = compute_metrics((logits, labels))
        assert result["accuracy"] == 1.0

    def test_returns_dict_with_accuracy_key(self):
        logits = np.array([[1.0, 0.0]])
        labels = np.array([0])
        result = compute_metrics((logits, labels))
        assert "accuracy" in result
        assert isinstance(result["accuracy"], float)
