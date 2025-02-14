# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/2/14 22:20
@File    : naive_bayes_numpy.py
@Description : 
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_priors = {}
        self.class_means = {}
        self.class_variances = {}
        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = X_c.shape[0] / float(X.shape[0])
            self.class_means[c] = X_c.mean(axis=0)
            self.class_variances[c] = X_c.var(axis=0)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.class_priors[c])
            class_conditional = np.sum(np.log(self._pdf(c, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.class_means[class_idx]
        var = self.class_variances[class_idx]
        numerator = np.exp(-(x - mean)**2 / (2*var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print(f"Predicted: {y_pred[:5]}")
    print(f"Actual: {y_test[:5]}")


if __name__ == "__main__":
    exit(main())
