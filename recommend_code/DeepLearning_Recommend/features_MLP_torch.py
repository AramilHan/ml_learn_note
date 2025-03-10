# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/3/9 22:39
@File    : features_MLP_torch.py
@Description : 
"""
import os
import sys
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
from Logistic_Regression import dataloader4ml100kIndexs
from data_set import filepaths as fp

class features_MLP(nn.Module):
    def __init__(self, n_user_features, n_item_features, user_df, item_df, dim):
        super(features_MLP, self).__init__()
        # 随机初始化用户特征的特征向量
        self.user_features = nn.Embedding(n_user_features, dim, max_norm=1)
        # 随机初始化物品特征的特征向量
        self.item_features = nn.Embedding(n_item_features, dim, max_norm=1)

        # 记录用户和物品的特征索引
        self.user_df = user_df
        self.item_df = item_df

        # 得到用户和物品特征种类数量之和
        total_features_num = user_df.shape[1] + item_df.shape[1]

        # 定义MLP传播的全连接层
        self.mlp_dense1 = self.dense_layer(dim * total_features_num, dim * total_features_num // 2)
        self.mlp_dense2 = self.dense_layer(dim * total_features_num // 2, dim)
        self.mlp_dense3 = self.dense_layer(dim, 1)

        self.sigmoid = nn.Sigmoid()

    def dense_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh()
        )

    def forward(self, u, i, is_train=None):
        user_ids = torch.LongTensor(self.user_df.loc[u].values)
        item_ids = torch.LongTensor(self.item_df.loc[i].values)
        # [batch_size, user_features_number, dim]
        user_features = self.user_features(user_ids)
        # [batch_size, item_features_number, dim]
        item_features = self.item_features(item_ids)

        # 将用户和物品的特征拼接起来
        # [batch_size, total_features_number, dim]
        uv = torch.cat([user_features, item_features], dim=1)

        # 将向量平铺以方便后续计算
        # [batch_size, total_features_number * dim]
        uv = uv.reshape((len(u), -1))

        # 开始MLP的传播
        # [batch_size, total_features_number * dim // 2]
        uv = self.mlp_dense1(uv)
        # [batch_size, dim]
        uv = self.mlp_dense2(uv)
        # 训练时采取dropout来防止过拟合
        if is_train:
            uv = F.dropout(uv)
        # [batch_size, 1]
        uv = self.mlp_dense3(uv)

        # [batch_size]
        uv = torch.squeeze(uv)
        logit = self.sigmoid(uv)
        return logit

# 做评估
def doEva(model, test_triples):
    d = torch.LongTensor(test_triples)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = model(u, i)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])

    precision = precision_score(r, y_pred, zero_division=0)
    recall = recall_score(r, y_pred, zero_division=0)
    accuracy = accuracy_score(r, y_pred)
    return precision, recall, accuracy

def train(epochs=10, batch_size=1024, lr=0.001, dim=256, eva_per_epochs=1):
    # 读取数据
    train_triples, test_triples, user_df, item_df, n_user_features, n_item_features = dataloader4ml100kIndexs.read_data_user_item_df()
    print(type(train_triples), type(test_triples), type(user_df), type(item_df))
    # 初始化模型
    model = features_MLP(n_user_features, n_item_features, user_df, item_df, dim)
    # 定义损失函数
    loss_fn = nn.BCELoss()
    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        all_loss = 0
        for u, i, r in DataLoader(train_triples, batch_size=batch_size, shuffle=True):
            r = torch.FloatTensor(r.detach().numpy())
            optimizer.zero_grad()
            logit = model(u, i, is_train=True)
            loss = loss_fn(logit, r)
            all_loss += loss
            loss.backward()
            optimizer.step()
        print(f'epoch: {e}, avg_loss: {all_loss / (len(train_triples) // batch_size):.4f}')

        if e % eva_per_epochs == 0:
            p, r, acc = doEva(model, train_triples)
            print(f"Train:  p: {p:.4f} | r: {r:.4f} | acc: {acc:.4f}")
            p, r, acc = doEva(model, test_triples)
            print(f"Test:  p: {p:.4f} | r: {r:.4f} | acc: {acc:.4f}")


def main():
    train()


if __name__ == "__main__":
    exit(main())
