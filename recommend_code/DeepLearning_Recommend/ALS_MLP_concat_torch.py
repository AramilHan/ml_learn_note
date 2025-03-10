# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/3/9 21:00
@File    : ALS_MLP_concat_torch.py
@Description : 
"""
import os
import sys
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, accuracy_score
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
from Collaborative_Filtering import dataloader
from data_set import filepaths as fp


class ALS_MLP(nn.Module):
    def __init__(self, n_users, n_items, dim):
        """
        :param n_users: 用户数量
        :param n_items: 物品数量
        :param dim: 向量维度
        """
        super(ALS_MLP, self).__init__()
        # 随机初始化用户的向量
        self.users = nn.Embedding(n_users, dim, max_norm=1)
        # 随机初始化物品的向量
        self.items = nn.Embedding(n_items, dim, max_norm=1)
        # 第一层的输入的维度是向量维度 * 2，因为用户与物品向量需要进行拼接
        self.dense_layer_1 = self.dense_layer(dim * 2, dim)
        self.dense_layer_2 = self.dense_layer(dim, dim // 2)
        # 最后一层的输出维度为1，该值经过sigmoid激活函数后输出
        self.dense_layer_3 = self.dense_layer(dim // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def dense_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh())

    def forward(self, u, v, is_train=True):
        """
        :param u: 用户索引id shape: [batch_size]
        :param v: 物品索引id shape: [batch_size]
        :return: 用户向量与物品向量的内积 shape: [batch_size]
        """
        # [batch_size, dim]
        u = self.users(u)
        v = self.items(v)
        # [batch_size, dim * 2]
        uv = torch.cat([u, v], dim=1)
        # [batch_size, dim]
        uv = self.dense_layer_1(uv)
        # [batch_size, dim // 2]
        uv = self.dense_layer_2(uv)
        if is_train:
            uv = F.dropout(uv)
        # [batch_size, 1]
        uv = self.dense_layer_3(uv)
        # [batch_size]
        uv = torch.squeeze(uv)
        logit = self.sigmoid(uv)
        return logit

def doEva(net, d):
    d = torch.LongTensor(d)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = net(u, i, False)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = r.detach().numpy()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, accuracy

def train(epochs=10, batch_size=1024, lr=0.001, dim=256, eva_per_epochs=1):
    """
    :param epochs: 迭代次数
    :param batch_size: 一批次的数量
    :param lr: 学习率
    :param dim: 用户和物品向量的维度
    :param eva_per_epochs: 设定每几次进行一次验证
    """
    # 读取数据
    user_set, item_set, train_set, test_set = dataloader.readRecData(fp.Ml_100K.RATING,
                                                                     test_ratio=0.1)
    # 初始化模型
    model = ALS_MLP(len(user_set), len(item_set), dim)
    # 定义损失函数
    loss_fn = nn.BCELoss()
    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        all_loss = 0
        # 按批次读取数据
        for u, i, r in DataLoader(train_set, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            r = torch.FloatTensor(r.detach().numpy())
            result = model(u, i, True)
            loss = loss_fn(result, r)
            all_loss += loss
            loss.backward()
            optimizer.step()
        print(f"epoch: {e}, loss: {all_loss / (len(train_set) // batch_size):.4f}")

        if e % eva_per_epochs == 0:
            p, r, acc = doEva(model, train_set)
            print(f"Train:  p: {p:.4f}, r: {r:.4f}, acc: {acc:.4f}")
            p, r, acc = doEva(model, test_set)
            print(f"Test:  p: {p:.4f}, r: {r:.4f}, acc: {acc:.4f}")

def main():
    train()


if __name__ == "__main__":
    exit(main())
