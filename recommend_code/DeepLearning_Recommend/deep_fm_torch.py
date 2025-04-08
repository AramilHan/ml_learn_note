# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/4/6 17:05
@File    : deep_fm_torch.py
@Description : 
"""
import os
import sys

import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import torch
from torch import nn
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
from Logistic_Regression import dataloader4ml100kIndexs


class DeepFM(nn.Module):
    def __init__(self, n_features, user_df, item_df, dim=128):
        super(DeepFM, self).__init__()
        # 随机初始化所有特征的特征向量
        self.features = nn.Embedding(n_features, dim, max_norm=1)
        # FM的一次项
        self.linear = nn.Embedding(n_features, 1)
        # 记录好用户和物品的特征索引
        self.user_df = user_df
        self.item_df = item_df
        # 得到用户和物品特征类别数量的和
        total_neighbours = user_df.shape[1] + item_df.shape[1]
        # 初始化MLP层
        self.mlp_layer = self.__mlp(dim * total_neighbours)

    def __first_fm(self, x):
        print(x.shape)
        print(self.linear(x).shape)
        linear_terms = torch.sum(self.linear(x), dim=1)
        return linear_terms

    def __fm_cross(self, feature_embs):
        # features_embs: [batch_size, n_features, dim]
        # [batch_size, dim]
        square_of_sum = torch.sum(feature_embs, dim=1) ** 2
        # [batch_size, dim]
        sum_of_square = torch.sum(feature_embs ** 2, dim=1)
        # [batch_size, dim]
        output = square_of_sum - sum_of_square
        # [batch_size, 1]
        output = torch.sum(output, dim=1, keepdim=True)
        output = 0.5 * output
        return torch.squeeze(output)

    def __mlp(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

    def deep(self, feature_embs):
        # feature_embs: [batch_size, n_features, dim]
        # [batch_size, total_neighbours * dim]
        feature_embs = feature_embs.reshape((feature_embs.shape[0], -1))
        # [batch_size, 1]
        output = self.mlp_layer(feature_embs)
        # [batch_size]
        return torch.squeeze(output)

    def __get_all_features(self, u, i):
        users = torch.LongTensor(self.user_df.loc[u].values)
        items = torch.LongTensor(self.item_df.loc[i].values)
        return torch.cat((users, items), dim=1)

    def forward(self, u, i):
        # 得到用户与物品组合起来后的特征索引
        all_feature_index = self.__get_all_features(u, i)
        # 取出特征向量
        all_feature_embs = self.features(all_feature_index)
        # [batch_size]
        # first_fm_out = self.__first_fm(all_feature_embs)
        # [batch_size]
        fm_out = self.__fm_cross(all_feature_embs)
        # [batch_size]
        deep_out = self.deep(all_feature_embs)
        out = torch.sigmoid(fm_out + deep_out)
        return out

def do_eva(net, test_triples):
    d = torch.LongTensor(test_triples)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = net(u, i)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    precision = precision_score(r, y_pred)
    recall = recall_score(r, y_pred)
    accuracy = accuracy_score(r, y_pred)
    return precision, recall, accuracy

def train(epochs=20, batch_size=1024, lr=0.02, dim=128, eva_per_epochs=1, need_eva=True):
    # 读取数据
    train_triples, test_triples, user_df, item_df, n_features = dataloader4ml100kIndexs.read_data()
    # 初始化模型
    net = DeepFM(n_features, user_df, item_df, dim=dim)
    # 定义损失函数
    loss_fn = nn.BCELoss()
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.3)
    for e in range(epochs):
        all_loss = 0
        for u, i, r in DataLoader(train_triples, batch_size=batch_size, shuffle=True):
            r = torch.FloatTensor(r.detach().numpy())
            optimizer.zero_grad()
            logit = net(u, i)
            loss = loss_fn(logit, r)
            all_loss += loss
            loss.backward()
            optimizer.step()
        print(f"epoch {e}, avg_loss: {all_loss / (len(train_triples) // batch_size):.4f}")

        if e % eva_per_epochs == 0 and need_eva:
            p, r, acc = do_eva(net, train_triples)
            print(f"Train   precision: {p:.4f}, recall: {r:.4f}, accuracy: {acc:.4f}")
            p, r, acc = do_eva(net, test_triples)
            print(f"Test    precision: {p:.4f}, recall: {r:.4f}, accuracy: {acc:.4f}")

def main():
    train()


if __name__ == "__main__":
    exit(main())
