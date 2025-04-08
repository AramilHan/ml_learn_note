# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/4/8 11:56
@File    : AFM_torch.py
@Description : 
"""
import os
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Parameter, init
from sklearn.metrics import precision_score, recall_score, accuracy_score

sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
from Logistic_Regression import dataloader4ml100kIndexs

class AFM(nn.Module):
    def __init__(self, n_features, user_df, item_df, k, t):
        super(AFM, self).__init__()
        # 随机初始化所有特征的特征向量
        self.features = nn.Embedding(n_features, k, max_norm=1)
        # 注意力计算中的线性层
        self.attention_linear = nn.Linear(k, t)
        # AFM公式中的h
        self.h = init.xavier_uniform_(Parameter(torch.empty(t, 1)))
        # AFM公式中的p
        self.p = init.xavier_uniform_(Parameter(torch.empty(k, 1)))
        # 记录用户和物品的特征索引
        self.user_df = user_df
        self.item_df = item_df

    # FM聚合
    def FM_aggregator(self, feature_embs):
        # feature_embs: [batch_size, n_features, k]
        # [batch_size, k]
        square_of_sum = torch.square(torch.sum(feature_embs, dim=1))
        # [batch_size, k]
        sum_of_square = torch.sum(torch.square(feature_embs), dim=1)
        # [batch_size, k]
        output = square_of_sum - sum_of_square
        return output

    def attention(self, embs):
        # embs: [batch_size, k]
        # [batch_size, t]
        embs = self.attention_linear(embs)
        # [batch_size, t]
        embs = torch.relu(embs)
        # [batch_size, 1]
        embs = torch.matmul(embs, self.h)
        # [batch_size, 1]
        atts = torch.softmax(embs, dim=1)
        return atts

    # 把用户和物品的特征合并起来
    def __get_all_features(self, u, i):
        users = torch.LongTensor(self.user_df.loc[u].values)
        items = torch.LongTensor(self.item_df.loc[i].values)
        all_features = torch.cat([users, items], dim=1)
        return all_features

    def forward(self, u, i):
        # 用户与物品组合起来后的特征索引
        all_feature_index = self.__get_all_features(u, i)
        # 取出特征向量
        all_feature_embs = self.features(all_feature_index)
        # FM聚合输出
        embs = self.FM_aggregator(all_feature_embs)
        # 得到注意力
        atts = self.attention(embs)
        # [batch_size, 1]
        output = torch.matmul(atts * embs, self.p)
        # [batch_size]
        output = torch.squeeze(output)
        # [batch_size]
        logit = torch.sigmoid(output)
        return logit

# 做评估
def do_eva(model, test_triples):
    d = torch.LongTensor(test_triples)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = model(u, i)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])

    precision = precision_score(r, y_pred)
    recall = recall_score(r, y_pred)
    accuracy = accuracy_score(r, y_pred)
    return precision, recall, accuracy

def train(epochs=20, batch_size=1024, lr=0.02, k=128, t=64, eva_per_epochs=1, need_eva=True):
    # 读取数据
    train_triples, test_triples, user_df, item_df, n_features = dataloader4ml100kIndexs.read_data()
    # 初始化模型
    model = AFM(n_features, user_df, item_df, k, t)
    # 定义损失函数
    loss_fn = nn.BCELoss()
    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.3)
    # 开始训练
    for e in range(epochs):
        all_loss = 0
        for u, i, r in DataLoader(train_triples, batch_size=batch_size, shuffle=True):
            r = torch.FloatTensor(r.detach().numpy())
            optimizer.zero_grad()
            logit = model(u, i)
            loss = loss_fn(logit, r)
            all_loss += loss
            loss.backward()
            optimizer.step()
        print(f"epoch  {e},  avg_loss:  {all_loss / (len(train_triples) // batch_size): .4f}")

        # 评估模型
        if e % eva_per_epochs == 0 and need_eva:
            p, r, acc = do_eva(model, train_triples)
            print(f"Train  p: {p: .4f}, r: {r: .4f}, acc: {acc: .4f}")
            p, r, acc = do_eva(model, test_triples)
            print(f"Test   p: {p: .4f}, r: {r: 4f}, acc: {acc: 4f}")

def main():
    train()


if __name__ == "__main__":
    exit(main())
