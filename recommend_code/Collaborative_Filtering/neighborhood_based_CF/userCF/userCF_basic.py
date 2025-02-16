# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/2/16 17:15
@File    : userCF_basic.py
@Description : 
"""
import os
import sys
sys.path.append(os.path.abspath('../../../'))
from Collaborative_Filtering import dataloader
from data_set import filepaths as fp
from tqdm import tqdm
from Collaborative_Filtering.basic_similarity_metric import basic_sim as b_sim
import collections


def getSet(triples):
    user_items = collections.defaultdict(set)
    for u, i, r in triples:
        if r == 1:
            user_items[u].add(i)
    return user_items

# knn算法
def knn4set(train_set, k , sim_method):
    """
    :param train_set: 训练集合
    :param k: 近邻数量
    :param sim_method: 相似度方法
    :return: {样本1: [近邻1, 近邻2, 近邻3]}
    """
    sims = {}
    # 两个for循环遍历训练集
    for e1 in tqdm(train_set):
        ulist = []  # 初始化一个列表来记录样本e1的近邻
        for e2 in train_set:
            # 如果两个样本相同则跳过
            if e1 == e2 or len(train_set[e1] & train_set[e2]) == 0:
                # 如果两个样本的交集为0也跳过
                sim = sim_method(train_set[e1], train_set[e2])
                ulist.append((e2, sim))
        # 排序后取前K的样本
        sims[e1] = [i[0] for i in sorted(ulist, key=lambda x: x[1], reverse=True)[:k]]
    return sims

# 得到基于相似用户的推荐列表
def get_recommendations_by_userCF(user_sims, user_o_set):
    """
    :param user_sims: 用户的近邻集：{样本1: [近邻1, 近邻2]}
    :param user_o_set: 用户的原本喜欢的物品结合: {用户1: {物品1, 物品2}}
    :return: 每个用户的推荐列表{用户1: [物品1, 物品2]}
    """
    recommendations = collections.defaultdict(set)
    for u in user_sims:
        for sim_u in user_sims[u]:
            # 将近邻用户喜爱的电影与自己观看过的电影去重后推荐给自己
            recommendations[u] |= (user_o_set[sim_u] - user_o_set[u])
    return recommendations

# 得到基于UserCF的推荐列表
def trainUserCF(user_items, sim_method, k=5):
    user_sims = knn4set(user_items, k, sim_method)
    recommendations = get_recommendations_by_userCF(user_sims, user_items)
    return recommendations

def main():
    _, _, train_set, test_set = dataloader.readRecData(fp.Ml_100K.RATING, test_ratio=0.1)
    user_items = getSet(train_set)
    recommendations_by_userCF = trainUserCF(user_items, b_sim.cos4set, k=5)
    print(recommendations_by_userCF)


if __name__ == "__main__":
    exit(main())
