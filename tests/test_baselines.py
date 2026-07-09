import numpy as np
import pandas as pd


class TestMajorityClassLogic:
    def test_majority_is_most_frequent(self):
        labels = pd.Series([0, 0, 0, 1, 1])
        majority = labels.mode()[0]
        assert majority == 0

    def test_majority_accuracy(self):
        labels = pd.Series([0, 0, 0, 1, 1])
        majority = labels.mode()[0]
        accuracy = (labels == majority).mean()
        assert accuracy == 0.6

    def test_balanced_labels(self):
        labels = pd.Series([0, 0, 1, 1])
        majority = labels.mode()[0]
        accuracy = (labels == majority).mean()
        assert accuracy == 0.5

    def test_all_same_label(self):
        labels = pd.Series([1, 1, 1, 1])
        majority = labels.mode()[0]
        accuracy = (labels == majority).mean()
        assert accuracy == 1.0


class TestTfidfLogic:
    def test_tfidf_shape(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = ["the cat sat", "a dog ran", "the cat ran"]
        vec = TfidfVectorizer(max_features=10)
        X = vec.fit_transform(texts)
        assert X.shape[0] == 3
        assert X.shape[1] <= 10

    def test_logistic_regression_fits(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        texts = ["the cat", "a dog", "the cats", "some dogs"] * 10
        labels = [0, 0, 1, 1] * 10
        vec = TfidfVectorizer(max_features=10)
        X = vec.fit_transform(texts)
        clf = LogisticRegression(random_state=42)
        clf.fit(X, labels)
        preds = clf.predict(X)
        assert len(preds) == len(labels)
