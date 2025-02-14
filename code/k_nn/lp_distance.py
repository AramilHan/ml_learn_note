# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/2/12 21:23
@File    : lp_distance.py
@Description : Lp距离的实现方式
"""

def calc_lp():
    """
    计算x1 = (1, 1), x2 = (5, 1), x3 = (4, 4)，x1的最近邻点
    :return:
    """
    def calc_lp_distance(p, x, y):
        distance_lp = 0
        for index, xi in enumerate(x):
            distance_lp += pow(abs(xi - y[index]), p)
        return round(pow(distance_lp, 1/p), 2)

    for p in (1, 2, 3, 4):
        x = [1, 1]
        distance = 0
        tag = 2
        for index, y in enumerate([[5, 1], [4, 4]]):
            if index == 0:
                distance = calc_lp_distance(p, x, y)
                continue
            distance1 = calc_lp_distance(p, x, y)
            distance = min(distance, distance1)
            tag = 3 if distance == distance1 else 2
        print(f"当p={p}时，x{tag}是x1的最近邻点，距离为{distance}")

def main():
    calc_lp()
    pass
   
if __name__ == "__main__":
    exit(main())