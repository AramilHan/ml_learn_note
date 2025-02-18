# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/2/18 21:11
@File    : neighbor_sim.py
@Description : 
"""
import math

# 计算流行度
def getPopularity(data_sets):
    """
    :param data_sets: 用户或物品集合{iid: {uid1, uid2}}
    :return: 返回一个记录流行度的字典{iid1: ppl1, iid1: ppl2}
    """
    p = dict()
    for id in data_sets:
        frequency = len(data_sets[id])
        ppl = math.log1p(frequency)  # ln(1 + x)
        p[id] = ppl
    return p

# cos相似度考虑流行度ppl
def getIIFSim(s1, s2, populations):
    """
    :param s1: 用户或物品的集合
    :param s2: 用户或物品的集合
    :param populations: 流行度字典
    :return: IIF相似度
    """
    s = 0
    for i in s1 & s2:
        s += 1/populations[i]
    return s / (len(s1) * len(s2)) ** 0.5

# 归一化
def normalizePopularities(popularities):
    """
    :param popularities: 流行度字典
    :return: 归一化后的流行度字典
    """
    maxp = max(popularities.values())
    norm_ppl = {}
    for k in popularities:
        norm_ppl[k] = popularities[k] / maxp
    return norm_ppl

# alpha相似度
def getAlphaSim(s1, s2, norm_ppl1):
    """
    :param s1: 用户或物品集合
    :param s2: 用户或物品集合
    :param norm_ppl1: 归一化后的流形度字典
    :return: alpha相似度
    """
    alpha = (1 + norm_ppl1) / 2
    return len(s1 & s2) / (len(s1) ** (1-alpha) * len(s2) ** alpha)


def sigmoid(x):
    return 1/(1 + math.e ** (-x))


def main():
    pass


if __name__ == "__main__":
    exit(main())
