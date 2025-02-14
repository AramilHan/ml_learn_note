# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/2/13 12:12
@File    : cross_validate_sk_learn_k.py
@Description : 
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from knn_model import KNN

def cross_validate_knn(X, y, k_values, n_splits=5):
    # 定义交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_k = None
    best_accuracy = 0

    for k in k_values:
        accuracies = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            knn = KNN(k=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_k = k
    return best_k, best_accuracy

def main():
    # 加载Iris数据集
    iris = load_iris()
    X, y = iris.data, iris.target

    # 定义要测试的k值范围
    k_values = range(1, 21)
    # 使用交叉验证选择最优k值
    best_k, best_accuracy = cross_validate_knn(X, y, k_values)
    print(f"best k: {best_k}, best accuracy: {best_accuracy: .2f}")

    # 使用最优k值训练最终模型并评估
    knn_final = KNN(k=best_k)
    knn_final.fit(X, y)
    y_pred = knn_final.predict(X)
    final_accuracy = accuracy_score(y, y_pred)
    print(f"Final Model Accuracy on Full Dataset: {final_accuracy: .2f}")

if __name__ == "__main__":
    exit(main())
