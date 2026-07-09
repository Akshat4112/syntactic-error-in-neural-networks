import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Baselines:
    def __init__(self):
        self.df_train = pd.read_csv(PROJECT_ROOT / 'data' / 'train_df.csv')
        self.df_test = pd.read_csv(PROJECT_ROOT / 'data' / 'test_df.csv')
        self.results_dir = PROJECT_ROOT / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def majority_class(self):
        majority_label = self.df_train['labels'].mode()[0]
        train_acc = (self.df_train['labels'] == majority_label).mean()
        test_acc = (self.df_test['labels'] == majority_label).mean()
        print(f"Majority class label: {majority_label}")
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy:  {test_acc:.4f}")
        results = {
            'model': 'majority_class',
            'majority_label': int(majority_label),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
        }
        with open(self.results_dir / 'majority_class_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        return results

    def logistic_regression(self, max_features=10000):
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train = vectorizer.fit_transform(self.df_train['text'])
        X_test = vectorizer.transform(self.df_test['text'])
        y_train = self.df_train['labels'].values
        y_test = self.df_test['labels'].values

        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Logistic Regression (TF-IDF, {max_features} features)")
        print(f"  5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        clf.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        test_preds = clf.predict(X_test)
        report = classification_report(y_test, test_preds, output_dict=True)
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy:  {test_acc:.4f}")

        results = {
            'model': 'logistic_regression',
            'max_features': max_features,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'classification_report': report,
        }
        with open(self.results_dir / 'logistic_regression_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        return results

    def run_all(self):
        print("=" * 60)
        print("BASELINE MODELS")
        print("=" * 60)
        print()
        maj = self.majority_class()
        print()
        lr = self.logistic_regression()
        print()
        print("=" * 60)
        print("BASELINE SUMMARY")
        print("=" * 60)
        print(f"  Majority class:       {maj['test_accuracy']:.4f}")
        print(f"  Logistic regression:  {lr['test_accuracy']:.4f}")
        return {'majority_class': maj, 'logistic_regression': lr}
