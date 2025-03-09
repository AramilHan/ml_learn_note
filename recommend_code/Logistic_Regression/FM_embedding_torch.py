# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/3/9 13:47
@File    : FM_embedding_torch.py
@Description : 
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import dataloader4ml100kIndexs
from torch.utils.data import DataLoader
import torch
from torch import nn

class FM(nn.Module):
    def __init__(self, n_features, dim=128):
        super(FM, self).__init__()
        # 随机初始化所有特征的特征向量
        self.features = nn.Embedding(n_features, dim, max_norm=1)

    def FMcross(self, features_embs):
        # features_embs: [batch_size, n_features, dim]
        # [batch_size, dim]
        square_of_sum = torch.sum(features_embs, dim=1) ** 2
        # [batch_size, dim]
        sum_of_square = torch.sum(features_embs ** 2, dim=1)

        # [batch_size, dim]
        output = square_of_sum - sum_of_square
        # [batch_size, 1]
        output = torch.sum(output, dim=1, keepdim=True)
        output = 0.5 * output
        # [batch_size]
        return torch.squeeze(output)

    def __getAllFeatures(self, u, i, user_df, item_df):
        # 把用户和物品的特征合并起来
        users = torch.LongTensor(user_df.loc[u].values)
        items = torch.LongTensor(item_df.loc[i].values)
        all = torch.cat([users, items], dim=1)
        return all

    def forward(self, u, i, user_df, item_df):
        # 得到用户和物品组合起来后的特征索引
        all_feature_index = self.__getAllFeatures(u, i, user_df, item_df)
        # 取出特征向量
        all_feature_embs = self.features(all_feature_index)
        # 通过FM层得到输出
        out = self.FMcross(all_feature_embs)
        logits = torch.sigmoid(out)
        return logits

# 做评估
def doEva(net, test_triples, user_df, item_df):
    d = torch.LongTensor(test_triples)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = net(u, i, user_df, item_df)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = r
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc

def train(epochs=20, batch_size=1024, lr=0.01, dim=128, eva_per_epochs=1):
    # 读取数据
    train_triples, test_triples, user_df, item_df, n_features = dataloader4ml100kIndexs.read_data()
    # 初始化模型
    net = FM(n_features, dim)
    # 定义损失函数
    loss_fn = nn.BCELoss()
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        all_loss = 0
        for u, i, r in DataLoader(train_triples, batch_size=batch_size, shuffle=True):
            r = torch.FloatTensor(r.detach().numpy())
            optimizer.zero_grad()
            logits = net(u, i, user_df, item_df)
            loss = loss_fn(logits, r)
            all_loss += loss
            loss.backward()
            optimizer.step()
        print(f"epoch: {e}, avg_loss: {all_loss / (len(train_triples) // batch_size)}")

        if e % eva_per_epochs == 0:
            p, r, acc = doEva(net, train_triples, user_df, item_df)
            print(f"Train -> p: {p:.4f} | r: {r:.4f} | acc: {acc:.4f}")
            p, r, acc = doEva(net, test_triples, user_df, item_df)
            print(f"Test -> p: {p:.4f} | r: {r:.4f} | acc: {acc:.4f}")


def main():
    train()


if __name__ == "__main__":
    exit(main())
