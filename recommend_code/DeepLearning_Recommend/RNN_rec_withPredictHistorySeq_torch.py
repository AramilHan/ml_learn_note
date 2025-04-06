# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/4/2 21:18
@File    : RNN_rec_withPredictHistorySeq_torch.py
@Description : 联合RNN训练
"""
import os
import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
import RNN_data_prepare as dp
from data_set import filepaths as fp


class RNNRec(nn.Module):
    def __init__(self, n_items, dim=128):
        super(RNNRec, self).__init__()
        self.n_items = n_items
        # 随机初始化所有物品的物品向量
        self.items = nn.Embedding(n_items, dim, max_norm=1)
        self.rnn = nn.RNN(dim, dim, batch_first=True)

        # 初始化历史序列预测的全连接层及损失函数等
        self.ph_dense = self.dense_layer(dim, n_items)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # 初始化推荐预测损失函数等
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.BCELoss = nn.BCELoss()

    # 全连接层
    def dense_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Softmax()
        )

    # 历史物品序列预测的前向传播
    def forward_pred_history(self, outs, history_seqs):
        # outs [batch_size, len_seqs, dim]
        # [batch_size, len_seqs, n_items]
        outs = self.ph_dense(outs)
        # [batch_size, len_seqs, n_items]
        outs = self.softmax(outs)
        # [batch_size * len_seqs, n_items]
        outs = outs.reshape(-1, self.n_items)
        # [batch_size * len_seqs]
        history_seqs = history_seqs.reshape(-1)
        return self.cross_entropy_loss(outs, history_seqs)

    # 推荐CTR预测的前向传播
    def forward_rec(self, h, item, y):
        # h: [1, batch_size, dim]
        # [batch_size, hidden_size]
        h = torch.squeeze(h)
        # [batch_size, dim]
        one_item = self.items(item)
        # [batch_size]
        out = torch.sum(h * one_item, dim=1)
        # [batch_size]
        logit = self.sigmoid(out)
        return self.BCELoss(logit, y)

    # 整体前向传播
    def forward(self, x, history_seqs, item, y):
        """
        :param x: 输入序列
        :param history_seqs: 要预测的序列，其实就是与x错开一位的历史记录
        :param item: 候选物品
        :param y: 0或1的标注
        :return: 联合训练的总损失函数值
        """
        # [batch_size, len_seqs, dim]
        item_embs = self.items(x)
        outs, h = self.rnn(item_embs)
        hp_loss = self.forward_pred_history(outs, history_seqs)
        rec_loss = self.forward_rec(h, item, y)
        return hp_loss + rec_loss

    # 因为模型forward函数输出的是损失函数值，所以需要一个预测函数方便预测与评估
    def predict(self, x, item):
        item_embs = self.items(x)
        _, h = self.rnn(item_embs)
        h = torch.squeeze(h)
        one_items = self.items(item)
        out = torch.sum(h * one_items, dim=1)
        logit = self.sigmoid(out)
        return logit

# 做评估
def do_eva(net, test_triple):
    d = torch.LongTensor(test_triple)
    x = d[:, :-2]
    item = d[:, -2]
    y = d[:, -1].float()

    with torch.no_grad():
        out = net.predict(x, item)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    return precision, recall, accuracy

def train(epochs=10, batch_size=1024, lr=0.001, dim=128, eva_per_epochs=1):
    # 读取数据
    train_data, test_data, all_items = dp.getTrainAndTestSeqs(fp.Ml_latest_small.SEQS)
    # 初始化模型
    net = RNNRec(max(all_items)+1, dim)
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        all_loss = 0
        for seq in tqdm(DataLoader(train_data, batch_size=batch_size, shuffle=True)):
            x = torch.LongTensor(seq[:, :-2].detach().numpy())
            history_seqs = torch.LongTensor(seq[:, 1: -1].detach().numpy())
            item = torch.LongTensor(seq[:, -2].detach().numpy())
            y = torch.FloatTensor(seq[:, -1].detach().numpy())
            optimizer.zero_grad()
            loss = net(x, history_seqs, item, y)
            all_loss += loss
            loss.backward()
            optimizer.step()
        avg_loss = all_loss / (len(train_data) // batch_size)
        print(f"epoch: {e}, avg_loss: {avg_loss: .4f}")

        # 评估模型
        if e % eva_per_epochs == 0:
            p, r, acc = do_eva(net, train_data)
            print(f"Train  p: {p}, r: {r}, acc: {acc}")
            p, r, acc = do_eva(net, test_data)
            print(f"Test  p: {p}, r: {r}, acc: {acc}")


def main():
    train()


if __name__ == "__main__":
    exit(main())
