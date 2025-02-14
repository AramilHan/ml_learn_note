# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/2/14 10:10
@File    : kdtree_sklearn.py
@Description : 
"""
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width


def plot_kdtree(points, query_point, nearest_neighbors):
    flg, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], c="blue", label="Data Points")
    ax.scatter(query_point[0, 0], query_point[0, 1], c="red", marker="x", s=100, label="Query Point")
    ax.scatter(nearest_neighbors[0, 0], nearest_neighbors[0, 1], c="green", marker="o",facecolors="none", edgecolors="g", s=100, label="Nearest Neighbors")
    ax.plot([query_point[0, 0], nearest_neighbors[0,0]],
            [query_point[0, 1], nearest_neighbors[0, 1]],
            "k--", linewidth=1)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("K-D-Tree")
    ax.legend()
    plt.grid(True)
    plt.show()

def main():
    points = np.array([
        [2, 3],
        [5, 4],
        [9, 6],
        [4, 7],
        [8, 1],
        [7, 2]
    ])
    # 构建K-D树
    kdtree = KDTree(points)
    # 查询最近邻
    query_point = np.array([[5, 5]])
    distances, indices = kdtree.query(query_point, k=1)
    print(f"query_point: {query_point.flatten()}")
    print(f"nearest neighbors: {points[indices][0]}")
    print(f"distance to nearest neighbor: {distances[0][0]}")
    print(points[indices])
    plot_kdtree(points, query_point, points[indices][0])


if __name__ == "__main__":
    exit(main())
