# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/4/2 14:19
@File    : RNN_rec_ALS_torch.py
@Description :
"""
import os
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
import RNN_data_prepare as dp
from data_set import filepaths as fp

class RNNALSRec(nn.Module):
    def __init__(self, n_items, dim=128):
        super(RNNALSRec, self).__init__()
        # 随机初始化所有物品的特征向量
        self.items = nn.Embedding(n_items, dim, max_norm=1)
        # 因为要进行向量点积运算，所以RNN层的输出向量维度也需要与物品向量一致
        self.rnn = nn.RNN(dim, dim, batch_first=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, item):
        # [batch_size, len_seqs, dim]
        item_embs = self.items(x)
        # [1, batch_size, dim]
        _, h = self.rnn(item_embs)
        # [batch_size, dim]
        h = torch.squeeze(h)
        # [batch_size, dim]
        one_item = self.items(item)
        # [batch_size]
        out = torch.sum(h * one_item, dim=1)
        logit = self.sigmoid(out)
        return logit


# 做评估
def doEva(net, test_triple):
    d = torch.LongTensor(test_triple)
    x = d[:, :-2]
    item = d[:, -2]
    y = d[:, -1].float()

    with torch.no_grad():
        out = net(x, item)

    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    return precision, recall, accuracy

def train(epochs=10, batch_size=1024, lr=0.001, dim=128, eva_per_epochs=1):
    # 读取数据
    train_data, test_data, all_items = dp.getTrainAndTestSeqs(fp.Ml_latest_small.SEQS)
    # 初始化模型
    net = RNNALSRec(max(all_items) + 1, dim)
    # 定义损失函数
    loss_fn = nn.BCELoss()
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        all_loss = 0
        for seq in DataLoader(train_data, batch_size=batch_size, shuffle=True):
            x = torch.LongTensor(seq[:, :-2].detach().numpy())
            item = torch.LongTensor(seq[:, -2].detach().numpy())
            y = torch.FloatTensor(seq[:, -1].detach().numpy())

            optimizer.zero_grad()
            y_pred = net(x, item)
            loss = loss_fn(y_pred, y)
            all_loss += loss
            loss.backward()
            optimizer.step()

        avg_loss = all_loss / (len(train_data) // batch_size)
        print(f'epoch {e}, avg_loss={avg_loss:.4f}')

        # 评估模型
        if e % eva_per_epochs != 0:
            continue

        p, r, acc = doEva(net, train_data)
        print(f'Train   p: {p:.4f}, r: {r:.4f}, acc: {acc:.4f}')
        p, r, acc = doEva(net, test_data)
        print(f"Test    p: {p:.4f}, r: {r:.4f}, acc: {acc:.4f}")

def main():
    train()


if __name__ == "__main__":
    exit(main())
