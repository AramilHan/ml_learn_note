# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/2/14 22:44
@File    : naive_bayes_scikit_learn.py
@Description : 
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    X = X[:, :2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    # 打印分类报告
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    # 打印混淆矩阵
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    # 可视化决策边界
    plt.figure(figsize=(10, 6))
    plot_decision_regions(X, y, clf=gnb, legend=2)
    plt.title("Decision Boundaries of Gaussian Naive Bayes")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    exit(main())
