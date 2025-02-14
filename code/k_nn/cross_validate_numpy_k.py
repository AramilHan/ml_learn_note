# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/2/13 11:55
@File    : cross_validate_numpy_k.py
@Description : 
"""
import numpy as np
from sklearn.datasets import load_iris
from knn_model import KNN

def cross_validate_knn(X, y, k_values, n_splits=5):
    num_samples = len(X)
    fold_size = num_samples // n_splits
    best_k = None
    best_accuracy = 0

    for k in k_values:
        # 准确率合集
        accuracies = []
        for i in range(n_splits):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i != n_splits - 1 else num_samples

            X_test_fold = X[start_idx: end_idx]
            y_test_fold = y[start_idx: end_idx]

            X_train_fold = np.concatenate((X[:start_idx], X[end_idx:]), axis=0)
            y_train_fold = np.concatenate((y[:start_idx], y[end_idx:]), axis=0)

            knn = KNN(k=k)
            knn.fit(X_train_fold, y_train_fold)
            y_pred_fold = knn.predict(X_test_fold)
            accuracy = np.mean(y_pred_fold == y_test_fold)
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_k = k
    return best_k, best_accuracy

def main():
    # 加载iris数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    # 定义测试的k值范围
    k_values = range(1, 21)
    # 使用交叉验证选择最优k值
    best_k, best_accuracy = cross_validate_knn(X, y, k_values)
    print(f"best k: {best_k}, best accuracy: {best_accuracy: .2f}")

    # 使用最优k值训练最终模型并评估
    knn_final = KNN(k=best_k)
    knn_final.fit(X, y)
    y_pred = knn_final.predict(X)
    final_accuracy = np.mean(y_pred == y)
    print(f"final model accuracy on full dataset: {final_accuracy:.2f}")

if __name__ == "__main__":
    exit(main())
