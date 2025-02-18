# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/2/18 00:01
@File    : evaluate.py
@Description : 
"""
def precision4Set(test_pos_set, test_neg_set, pred_set):
    """
    :param test_pos_set: 真实的用户喜爱的物品集合{iid1, iid2, iid3}
    :param test_neg_set: 真实的用户不喜爱的物品集合{iid1, iid2, iid3}
    :param pred_set: 预测的推荐集合{iid1, iid2, iid3}
    :return: 精准率
    """
    TP = len(pred_set & test_pos_set)
    FP = len(pred_set & test_neg_set)
    p = TP / (TP + FP) if (TP + FP) > 0 else None
    # p = TP/len(pred_set)
    return p

def recall4Set(test_set, pred_set):
    """
    :param test_set: 真实的用户喜爱的物品集合{iid1, iid2, iid3}
    :param pred_set: 预测的推荐集合{iid1, iid2, iid3}
    :return: 召回率
    """
    # 计算它们的交集数量 除以 测试集的数量
    return len(pred_set & test_set) / len(test_set)
