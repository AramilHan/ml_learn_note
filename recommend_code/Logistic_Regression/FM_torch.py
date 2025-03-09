# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/3/8 22:42
@File    : FM_torch.py
@Description : 
"""
import torch
from torch import nn
from torch.nn import Parameter, init
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import dataloader4ml100kOneHot


class FM(nn.Module):
    def __init__(self, n_features, dim):
        """
        :param n_features: 特征数量
        :param dim: 隐向量维度
        """
        super(FM, self).__init__()
        self.w0 = init.xavier_uniform_(Parameter(torch.empty(1, 1)))
        self.w1 = init.xavier_uniform_(Parameter(torch.empty(n_features, 1)))
        self.w2 = init.xavier_uniform_(Parameter(torch.empty(n_features, dim)))

    def FM_cross(self, x):
        """
        FM交叉相乘
        """
        # [batch_size, dim]
        square_of_sum = torch.matmul(x, self.w2) ** 2
        # [batch_size, dim]
        sum_of_square = torch.matmul(x ** 2, self.w2 ** 2)

        output = square_of_sum - sum_of_square
        output = torch.sum(output, dim=1, keepdim=True)
        output = 0.5 * output
        return output

    def forward(self, x):
        lr_out = self.w0 + torch.matmul(x, self.w1)
        cross_out = self.FM_cross(x)
        logits = torch.sigmoid(lr_out + cross_out)
        return logits


# 做评估
def doEva(net, x, y):
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    with torch.no_grad():
        out = net(x)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = y
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc

def train(epochs=20, batch_size=1024, lr=0.01, dim=128, eva_per_epochs=1):
    # 读取数据
    x_train, x_test, y_train, y_test = dataloader4ml100kOneHot.read_data()
    # 得到特征数量
    features = len(x_train[0])
    # 初始化模型
    net = FM(features, dim)
    # 定义损失函数
    loss_fn = nn.BCELoss()
    # 定义优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    # 初始化数据迭代器
    dataIter = dataloader4ml100kOneHot.DataIter(x_train, y_train)
    # 开始训练
    for epoch in range(epochs):
        all_loss = 0
        for datas in dataIter.iter(batch_size=batch_size):
            Xs = torch.FloatTensor([d[0] for d in datas])
            labels = torch.FloatTensor([d[1] for d in datas])
            optimizer.zero_grad()
            logits = net(Xs)
            loss = loss_fn(logits, labels)
            all_loss += loss
            loss.backward()
            optimizer.step()
        print(f"epoch: {epoch}, avg_loss: {all_loss / (len(y_train) // batch_size):.4f}")

        # 评估模型
        if epoch % eva_per_epochs == 0:
            p, r, acc = doEva(net, x_train, y_train)
            print(f"Train  p: {p:.4f} | r: {r:.4f} | acc: {acc:.4f}")
            p, r, acc = doEva(net, x_test, y_test)
            print(f"Test  p: {p:.4f} | r: {r:.4f} | acc: {acc:.4f}")


def main():
    train()


if __name__ == "__main__":
    exit(main())
