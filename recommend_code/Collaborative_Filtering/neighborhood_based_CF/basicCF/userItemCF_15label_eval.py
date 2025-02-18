# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/2/18 12:14
@File    : userItemCF_15label_eval.py
@Description : 
"""
import os
import sys

import numpy as np

sys.path.append(os.path.abspath("../../../"))
from Collaborative_Filtering import dataloader
from data_set import filepaths as fp
from tqdm import tqdm
from Collaborative_Filtering.basic_similarity_metric import basic_sim as b_sim
from utils import evaluate
import collections

# 字典形式读取数据，返回{uid1: {iid1: rate, iid2: rate}}
def getDict(triples):
    user_items = collections.defaultdict(dict)
    item_users = collections.defaultdict(dict)
    for u, i, r in triples:
        user_items[u][i] = r
        item_users[i][u] = r
    return user_items, item_users

# 集合形式读取数据，评分大于4的为正例，反之为反例。
def getPosAndNegSet(triples):
    user_pos_items = collections.defaultdict(set)
    user_neg_items = collections.defaultdict(set)
    for u, i, r in triples:
        if r >= 4:
            user_pos_items[u].add(i)
        else:
            user_neg_items[u].add(i)
    return user_pos_items, user_neg_items

# 根据评分字典得到cos相似度
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
        return 0  # 如果没有交集，则相似度一定为0
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
def get_recommendations_by_userCF(user_sims, user_o_set, user_pos_items_train):
    recommendations = collections.defaultdict(set)
    for u in user_sims:
        for sim_u in user_sims[u]:
            recommendations[u] |= (user_pos_items_train[sim_u] - set(user_o_set[u].keys()))
    return recommendations

# 得到基于相似物品的推荐列表
def get_recommendations_by_itemCF(item_sims, user_o_set):
    recommendations = collections.defaultdict(set)
    for u in user_o_set:
        for item in user_o_set[u]:
            recommendations[u] |= set(item_sims[item]) - set(user_o_set[u].keys())
    return recommendations

def trainUserCF(user_items_train, user_pos_items_train, k=5):
    user_sims = knn4Dict(user_items_train, k)
    recommendations = get_recommendations_by_userCF(user_sims, user_items_train, user_pos_items_train)
    return recommendations

def trainItemCF(user_items_train, item_users_train, k=5):
    item_sims = knn4Dict(item_users_train, k)
    recommendations = get_recommendations_by_itemCF(item_sims, user_items_train)
    return recommendations

def evaluation(test_set, user_neg_items, pred_set):
    total_r = 0.0
    total_p = 0.0
    has_p_count = 0
    for uid in test_set:
        if len(test_set[uid]) == 0:
            continue
        total_r += evaluate.recall4Set(test_set[uid], pred_set[uid])
        p = evaluate.precision4Set(test_set[uid], user_neg_items[uid], pred_set[uid])
        if not p:
            continue
        total_p += p
        has_p_count += 1
    print(f"Precision: {total_p / has_p_count:.4f} | Recall: {total_r / has_p_count:.4f}")

def main():
    _, _, train_set, test_set = dataloader.readRecData(fp.Ml_100K.RATING5, test_ratio=0.1)
    user_items_train, item_users_train = getDict(train_set)
    user_pos_items_train, _ = getPosAndNegSet(train_set)
    # 测试集，正例集合与负例集合
    user_pos_items_test, user_neg_items_test = getPosAndNegSet(test_set)

    recommendations_by_userCF = trainUserCF(user_items_train, user_pos_items_train, k=5)
    recommendations_by_itemCF = trainItemCF(user_items_train, item_users_train, k=5)
    print("UserCF")
    evaluation(user_pos_items_test, user_neg_items_test, recommendations_by_userCF)
    print("ItemCF")
    evaluation(user_pos_items_test, user_neg_items_test, recommendations_by_itemCF)

if __name__ == "__main__":
    exit(main())
