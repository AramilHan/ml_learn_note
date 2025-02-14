# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/2/13 12:13
@File    : knn_model.py
@Description : 
"""
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # 计算x与训练集X之间所有点的距离
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # 按x与各个点之间的距离排序，选取前k个点
        k_indices = np.argsort(distances)[:self.k]
        # 获取k个点的标签
        k_nearest_labels = self.y_train[k_indices]
        # 获取k个点内标签最多的标签
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
