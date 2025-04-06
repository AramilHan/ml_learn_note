# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/3/10 23:23
@File    : RNN_rec_torch.py
@Description : 
"""
import os
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
import RNN_data_prepare as dp
from data_set import filepaths as fp

class RNNRec(nn.Module):
    def __init__(self, n_items, hidden_size=64, dim=128):
        super(RNNRec, self).__init__()
        # 随机初始化物品向量
        self.item_embeddings = nn.Embedding(n_items, dim, max_norm=1)
        self.rnn = nn.RNN(dim, hidden_size, batch_first=True)
        self.dense1 = self.dense_layer(hidden_size, hidden_size // 2)
        self.dense2 = self.dense_layer(hidden_size // 2, 1)

    def dense_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh()
        )

    def forward(self, x, is_train=True):
        # [batch_size, len_seqs, dim]
        item_embedding = self.item_embeddings(x)
        # [1, batch_size, hidden_size]
        _, hn = self.rnn(item_embedding)
        # [batch_size, hidden_size]
        hn = torch.squeeze(hn)
        if is_train:
            hn = F.dropout(hn)
        # [batch_size, 1]
        out = self.dense1(hn)
        out = self.dense2(out)
        # [batch_size]
        out = torch.squeeze(out)
        out = torch.sigmoid(out)
        return out

def doEva(model, test_triples):
    d = torch.LongTensor(test_triples)
    x = d[:, :-1]
    y = d[:, -1].float()
    with torch.no_grad():
        out = model(x, is_train=False)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])

    p = precision_score(y, y_pred)
    r = recall_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    return p, r, acc

def train(epochs=10, batch_size=1024, lr=0.001, rnn_hidden_size=64, dim=128, eva_per_epochs=1):
    # 读取数据
    train_infos, test_infos, all_items = dp.getTrainAndTestSeqs(fp.Ml_latest_small.SEQS)
    # 初始化模型
    model = RNNRec(max(all_items) + 1, rnn_hidden_size, dim)
    # 定义损失函数
    loss_fn = nn.BCELoss()
    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        all_loss = 0
        for seq in DataLoader(train_infos, batch_size=batch_size, shuffle=True):
            x = torch.LongTensor(seq[:, :-1].detach().numpy())
            y = torch.FloatTensor(seq[:, -1].detach().numpy())
            optimizer.zero_grad()
            logit = model(x, is_train=True)
            loss = loss_fn(logit, y)
            all_loss += loss
            loss.backward()
            optimizer.step()
        print(f"epoch: {e} | loss: {all_loss / (len(train_infos) // batch_size):.4f}")

        if e % eva_per_epochs == 0:
            p, r, acc = doEva(model, train_infos)
            print(f"Train: p: {p:.4f}, r: {r:.4f}, acc: {acc:.4f}")
            p, r, acc = doEva(model, test_infos)
            print(f"Test: p: {p:.4f}, r: {r:.4f}, acc: {acc:.4f}")

def main():
    train()


if __name__ == "__main__":
    exit(main())
