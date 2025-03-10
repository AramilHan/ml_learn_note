# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/3/9 14:43
@File    : ALS_MLP_torch.py
@Description : 
"""
import os
import sys
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, accuracy_score
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

        # 初始化用户向量的隐层
        self.u_hidden_layer1 = self.dense_layer(dim, dim // 2)
        self.u_hidden_layer2 = self.dense_layer(dim // 2, dim // 4)

        # 初始化物品向量的隐层
        self.i_hidden_layer1 = self.dense_layer(dim, dim // 2)
        self.i_hidden_layer2 = self.dense_layer(dim // 2, dim // 4)

        self.sigmoid = nn.Sigmoid()

    def dense_layer(self, in_features, out_features):
        # 每一个MLP单元包含一个线性层和非线性激活层，本次代码激活层采用Tanh双曲正切函数
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh())

    def forward(self, u, v, is_train=True):
        """
        :param u: 用户索引id shape: [batch_size]
        :param v: 物品索引id shape: [batch_size]
        :return: 用户向量与物品向量的内积，shape: [batch_size]
        """
        u = self.users(u)
        v = self.items(v)

        u = self.u_hidden_layer1(u)
        u = self.u_hidden_layer2(u)

        v = self.i_hidden_layer1(v)
        v = self.i_hidden_layer2(v)

        # 训练时采用dropout来防止过拟合
        if is_train:
            u = F.dropout(u)
            v = F.dropout(v)

        uv = torch.sum(u * v, dim=1)
        logit = self.sigmoid(uv * 3)
        return logit

def doEva(net, d):
    d = torch.LongTensor(d)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = net(u, i, False)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = r.detach().numpy()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, accuracy

def train(epochs=10, batch_size=1024, lr=0.001, dim=256, eva_per_epochs=1):
    """
    :param epochs: 迭代次数
    :param batch_size: 一批次的数量
    :param lr: 学习率
    :param dim: 用户物品向量的维度
    :param eva_per_epochs: 设定每几次进行一次验证
    """
    # 读取数据
    user_set, item_set, train_set, test_set = dataloader.readRecData(fp.Ml_100K.RATING, test_ratio=0.1)
    # 初始化ALS模型
    net = ALS_MLP(len(user_set), len(item_set), dim)
    # 定义优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    # 定义损失函数
    loss_fn = nn.BCELoss()
    # 开始迭代
    for e in range(epochs):
        all_loss = 0
        # 每一批次的读取数据
        for u, i, r in DataLoader(train_set, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            r = torch.FloatTensor(r.detach().numpy())
            result = net(u, i)
            loss = loss_fn(result, r)
            all_loss += loss
            loss.backward()
            optimizer.step()
        print(f"epoch: {e}, avg_loss: {all_loss / (len(train_set) // batch_size):.4f}")

        # 评估模型
        if e % eva_per_epochs == 0:
            p, r, acc = doEva(net, train_set)
            print(f"Train: precision: {p:.4f} | recall: {r:.4f} | accuracy: {acc:.4f}")
            p, r, acc = doEva(net, test_set)
            print(f"Test: precision: {p:.4f} | recall: {r:.4f} | accuracy: {acc:.4f}")


def main():
    train()


if __name__ == "__main__":
    exit(main())
