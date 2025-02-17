# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/2/17 21:58
@File    : userItemCF_15label.py
@Description : 
"""
import os
import sys
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
from Collaborative_Filtering import dataloader
from data_set import filepaths as fp
from tqdm import tqdm
from Collaborative_Filtering.basic_similarity_metric import basic_sim as b_sim
from basicCF import userCF_basic as userCF
import collections
import numpy as np

# 字典形式读取数据，返回{uid: {iid1: rate, iid2: rate}}
def getDict(triples):
    user_items = collections.defaultdict(dict)
    item_users = collections.defaultdict(dict)
    for u, i, r in triples:
        user_items[u][i] = r
        item_users[i][u] = r
    return user_items, item_users

# 根据评分字典的到cos相似度
def getCosSimForDict(d1, d2):
    """
    :param d1: 字典{iid1: rate, iid2: rate}
    :param d2: 字典{iid2: rate, iid3: rate}
    :return: 得到cos相似度
    """
    s1 = set(d1.keys())
    s2 = set(d2.keys())
    inner = s1 & s2
    if len(inner) == 0:
        return 0  # 如果两个物品集没有相交，则相似度一定为0
    a1, a2 = [], []
    for i in inner:
        a1.append(d1[i])
        a2.append(d2[i])
    for i in s1 - inner:
        a1.append(d1[i])
        a2.append(0)
    for i in s2 - inner:
        a1.append(0)
        a2.append(d2[i])
    return b_sim.cos4vector(np.array(a1), np.array(a2))

# knn算法
def knn4Dict(train_set, k):
    sims = {}
    for e1 in tqdm(train_set):
        ulist = []
        for e2 in train_set:
            if e1 == e2:
                continue
            cosSim = getCosSimForDict(train_set[e1], train_set[e2])
            if cosSim != 0:
                ulist.append((e2, cosSim))
        sims[e1] = [i[0] for i in sorted(ulist, key=lambda x: x[1], reverse=True)[:k]]
    return sims

# 得到基于相似用户的推荐列表
def get_recommendations_by_userCF(user_sims, user_o_set, user_items):
    recommendations = collections.defaultdict(set)
    for u in user_sims:
        for sim_u in user_o_set[u]:
            recommendations[u] |= (user_items[sim_u] - set(user_o_set[u].keys()))
    return recommendations

# 得到基于相似物品的推荐列表
def get_recommendations_by_itemCF(item_sims, user_o_set):
    recommendations = collections.defaultdict(set)
    for u in user_o_set:
        for item in user_o_set[u]:
            recommendations[u] |= set(item_sims[item]) - set(user_o_set[u].keys())
    return recommendations

def trainUserCF(user_items_train, user_pos_items, k=5):
    user_sims = knn4Dict(user_items_train, k)
    recommendations = get_recommendations_by_userCF(user_sims, user_items_train, user_pos_items)
    return recommendations

def trainItemCF(user_items_train, item_users_train, k=5):
    item_sims = knn4Dict(item_users_train, k)
    recommendations = get_recommendations_by_itemCF(item_sims, user_items_train)
    return recommendations

def main():
    _, _, train_set, test_set = dataloader.readRecData(fp.Ml_100K.RATING5, test_ratio=0.1)
    user_items_train, item_users_train = getDict(train_set)
    user_pos_items = userCF.getSet(train_set)
    recommendations_by_userCF = trainUserCF(user_items_train, user_pos_items, k=5)
    print(recommendations_by_userCF)
    recommendations_by_itemCF = trainItemCF(user_items_train, item_users_train, k=5)
    print(recommendations_by_itemCF)


if __name__ == "__main__":
    exit(main())
