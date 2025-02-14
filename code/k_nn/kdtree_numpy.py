# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/2/13 22:20
@File    : kdtree_numpy.py
@Description : 
"""
import numpy as np

class KDNode:
    def __init__(self, point, left=None, right=None, axis=None):
        # 节点的点
        self.point = point
        # 左子树
        self.left = left
        # 右子树
        self.right = right
        # 切分轴
        self.axis = axis

def build_kdtree(points, depth=0):
    if len(points) == 0:
        return None

    # 根据当前深度选择轴
    k = points.shape[0]
    axis = depth % k
    # 对点列表进行排序并选择中位数作为轴元素
    sorted_points = points[np.argsort(points[:, axis])]
    median = len(sorted_points) // 2

    return KDNode(
        point=sorted_points[median],
        left=build_kdtree(sorted_points[:median], depth + 1),
        right=build_kdtree(sorted_points[median + 1:], depth + 1),
        axis=axis
    )

def find_nearest(node, target, depth=0, best=None):
    if node is None:
        return best
    # 将目标与当前节点沿分割轴的点进行比较
    axis = node.axis
    next_best = closest_point(best, node.point, target)
    print(f"axis: {axis}, next_best: {next_best}")
    # 遍历一颗子树
    if target[axis] < node.point[axis]:
        next_branch = node.left
        opposite_branch = node.right
    else:
        next_branch = node.right
        opposite_branch = node.left
    print(f"target[axis]: {target[axis]}, node.point[axis]: {node.point[axis]}")
    # 递归到下一个分支
    best = find_nearest(next_branch, target, depth + 1, next_best)
    # 检查是否需要检索其他分支
    if distance_squared(target, best) > (target[axis] - node.point[axis]) ** 2:
        best = find_nearest(opposite_branch, target, depth + 1, best)
    return best

def closest_point(p1, p2, target):
    if p1 is None:
        return p2
    elif p2 is None:
        return p1
    else:
        d1 = distance_squared(target, p1)
        d2 = distance_squared(target, p2)
        return p1 if d1 < d2 else p2

def distance_squared(point1, point2):
    return np.sum((point1 - point2) ** 2)

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
    root = build_kdtree(points)
    query_point = np.array([5, 5])
    nearest_neighbor = find_nearest(root, query_point)
    print(f"query_point: {query_point}")
    print(f"nearest_neighbor: {nearest_neighbor}")


if __name__ == "__main__":
    exit(main())
