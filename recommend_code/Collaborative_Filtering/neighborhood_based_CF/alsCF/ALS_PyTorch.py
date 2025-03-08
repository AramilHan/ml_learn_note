# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/3/1 13:29
@File    : ALS_PyTorch.py
@Description : 
"""
import os
import sys
import numpy as np
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
from data_set import filepaths as fp
from Collaborative_Filtering import dataloader
import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import precision_score, recall_score, accuracy_score


class ALS(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super(ALS, self).__init__()
        """
        :param n_users: 用户数量
        :param n_items: 物品数量
        :param dim: 向量维度
        """
        # 随机初始化用户的向量，将向量约束在L2范数为1以内
        self.users = nn.Embedding(n_users, dim, max_norm=1)
        # 随机初始化物品的向量，将向量约束在L2范数为1以内
        self.items = nn.Embedding(n_items, dim, max_norm=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, u, v):
        """
        前向传播，预测u，v的评分
        :param u: 用户索引id shape: [batch_size]
        :param v: 物品索引id shape: [batch_size]
        :return: 用户向量与物品向量的内积 shape: [batch_size]
        """
        u = self.users(u)
        v = self.items(v)
        uv = torch.sum(u * v, dim=1)
        logit = self.sigmoid(uv)
        return logit


def doEva(net, d):
    d = torch.LongTensor(d)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():  # 禁用梯度计算
        out = net(u, i)  # 前向传播时不计算梯度
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    r = torch.where(r==1, torch.tensor(1.), torch.tensor(0))
    y_true = r.detach().numpy()
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc


def train(epochs=10, batch_size=1024, lr=0.01, dim=64, eva_per_epochs=1):
    """
    :param epochs: 迭代次数
    :param batch_size: 一批次的数量
    :param lr: 学习率
    :param dim: 用户物品向量的维度
    :param eva_per_epochs: 设定每几次进行一次验证
    """
    # 读取数据
    user_set, item_set, train_set, test_set = dataloader.readRecData(
        fp.Ml_100K.RATING5, test_ratio=0.1)
    # 初始化ALS模型
    net = ALS(len(user_set), len(item_set), dim)
    # 定义优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    # 定义损失函数
    criterion = torch.nn.BCELoss()
    # 开始迭代
    for epoch in range(epochs):
        all_loss = 0
        # 每一批次地读取数据
        for u, i, r in DataLoader(train_set, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()  # 清空梯度
            r = torch.FloatTensor(r.detach().numpy())
            r = torch.where(r==1, torch.tensor(1.), torch.tensor(0.))
            result = net.forward(u, i)
            loss = criterion(result, r)
            all_loss += loss
            loss.backward()
            optimizer.step()
        print(f"epoch: {epoch}, avg_loss={all_loss/(len(train_set)//batch_size):.4f}")

        # 评估模型
        if epoch % eva_per_epochs == 0:
            p, r, acc = doEva(net, train_set)
            print(f"train: Precision {p:.4f} | Recall {r:.4f} | Accuracy {acc:.4f}")
            p, r, acc = doEva(net, test_set)
            print(f"test: Precision {p:.4f} | Recall {r:.4f} | Accuracy {acc:.4f}")


def main():
    train()


if __name__ == "__main__":
    exit(main())
