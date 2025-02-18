# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/2/18 00:22
@File    : userItemCF_01label_eval.py
@Description : 
"""
import os
import sys
sys.path.append(os.path.abspath('../../../'))
from Collaborative_Filtering import dataloader
from data_set import filepaths as fp
from tqdm import tqdm
from Collaborative_Filtering.basic_similarity_metric import basic_sim as b_sim
from utils import evaluate
import collections

# 集合形式读取数据，返回{uid1: {iid1, iid2, iid3}}
def getSet(triples):
    # 用户喜欢的物品集
    user_pos_items = collections.defaultdict(set)
    # 用户不喜欢的物品集
    user_neg_items = collections.defaultdict(set)
    # 用户交互过的所有物品集
    user_all_items = collections.defaultdict(set)
    # 已物品为索引，喜欢物品的用户集
    item_users = collections.defaultdict(set)
    for u, i, r in triples:
        user_all_items[u].add(i)
        if r == 1:
            user_pos_items[u].add(i)
            item_users[i].add(u)
        else:
            user_neg_items[u].add(i)
    return user_pos_items, item_users, user_neg_items, user_all_items

# knn算法
def knn4set(train_set, k, sim_method):
    """
    :param train_set: 训练集合
    :param k: 近邻数量
    :param sim_method: 相似度方法
    :return: {样本1: [近邻1, 近邻2, 近邻3]}
    """
    sims = {}
    # 两个for循环遍历训练集合
    for e1 in tqdm(train_set):
        ulist = []  # 初始化一个列表来记录样本e1的近邻
        for e2 in train_set:
            if e1 == e2 or len(train_set[e1] & train_set[e2]) == 0:
                continue
            # 用相似度方法取得两个样本的相似度
            sim = sim_method(train_set[e1], train_set[e2])
            ulist.append((e2, sim))
        # 排序后取前k个样本
        sims[e1] = [i[0] for i in sorted(ulist, key=lambda x: x[1], reverse=True)[:k]]
    return sims

# 得到基于相似用户的推荐列表
def get_recommendations_by_userCF(user_sims, user_o_set):
    """
    :param user_sims: 用户的近邻集
    :param user_o_set: 用户原本喜欢的物品集合
    :return: 每个用户的推荐列表
    """
    recommendations = collections.defaultdict(set)
    for u in user_sims:
        for sim_u in user_sims[u]:
            recommendations[u] |= (user_o_set[sim_u] - user_o_set[u])
    return recommendations

# 得到基于相似物品的推荐列表
def get_recommendations_by_itemCF(item_sims, user_o_set):
    """
    :param item_sims: 物品的近邻集
    :param user_o_set: 用户的原本喜欢的物品集合
    :return: 每个用户的推荐列表
    """
    recommendations = collections.defaultdict(set)
    for u in user_o_set:
        for item in user_o_set[u]:
            if item in item_sims:
                recommendations[u] |= set(item_sims[item]) - user_o_set[u]
    return recommendations

# 得到基于UserCF的推荐列表
def trainUserCF(user_items_train, sim_method, user_all_items, k=5):
    user_sims = knn4set(user_items_train, k, sim_method)
    recommendations = get_recommendations_by_userCF(user_sims, user_all_items)
    return recommendations

# 得到基于ItemCF的推荐列表
def trainItemCF(item_users_train, sim_method, user_all_items, k=5):
    item_sims = knn4set(item_users_train, k, sim_method)
    recommendations = get_recommendations_by_itemCF(item_sims, user_all_items)
    return recommendations

def evaluation(test_set, user_neg_items, pred_set):
    total_r = 0.0
    total_p = 0.0
    has_p_count = 0
    for uid in test_set:
        if len(test_set[uid]) == 0:
            continue
        p = evaluate.precision4Set(test_set[uid], user_neg_items[uid], pred_set[uid])
        total_r += evaluate.recall4Set(test_set[uid], pred_set[uid])
        if not p:
            continue
        total_p += p
        has_p_count += 1
    print(f"Precision:{total_p/has_p_count:.4f} | Recall:{total_r/has_p_count:.4f} ")

def main():
    _, _, train_set, test_set = dataloader.readRecData(fp.Ml_100K.RATING, test_ratio=0.1)
    user_items_train, item_users_train, _, user_all_items = getSet(train_set)
    user_pos_items_test, _, user_neg_items_test, _ = getSet(test_set)
    recommendations_by_userCF = trainUserCF(user_items_train, b_sim.cos4set, user_all_items, k=5)
    recommendations_by_itemCF = trainItemCF(item_users_train, b_sim.cos4set, user_all_items, k=5)
    print("UserCF")
    evaluation(user_pos_items_test, user_neg_items_test, recommendations_by_userCF)
    print("ItemCF")
    evaluation(user_pos_items_test, user_neg_items_test, recommendations_by_itemCF)

if __name__ == "__main__":
    exit(main())