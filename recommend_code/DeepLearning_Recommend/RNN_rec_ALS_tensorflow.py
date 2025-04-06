# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/4/2 15:23
@File    : RNN_rec_ALS_tensorflow.py
@Description : ALS联合RNN
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, constraints, Sequential
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
import RNN_data_prepare as dp
from data_set import filepaths as fp

class RNNALSRec(Model):
    def __init__(self, n_items, dim=128):
        super(RNNALSRec, self).__init__()
        # 随机初始化所有物品的特征向量
        self.items = layers.Embedding(n_items, dim, embeddings_constraint=tf.keras.constraints.MaxNorm(1))
        # 因为要进行向量点积运算，所以RNN层的输出向量维度也需要与物品向量一致
        self.rnn = layers.SimpleRNN(dim, return_sequences=False, return_state=False)
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x, item):
        # [batch_size, len_seqs, dim]
        item_embs = self.items(x)
        # [1, batch_size, dim]
        h = self.rnn(item_embs)
        # [batch_size, dim]
        h = tf.squeeze(h)
        # [batch_size, dim]
        one_item = self.items(item)
        # [batch_size]
        out = tf.reduce_sum(h * one_item, axis=1)
        logit = self.sigmoid(out)
        return logit

# 做评估
def doEva(model, test_triples):
    x = test_triples[:, :-2]
    item = test_triples[:, -2]
    y = test_triples[:, -1]
    out = model(x, item)
    y_pred = tf.where(out >= 0.5, 1., 0.)

    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    accuracy = tf.keras.metrics.Accuracy()

    precision.update_state(y_pred, y)
    recall.update_state(y_pred, y)
    accuracy.update_state(y_pred, y)
    return precision.result().numpy(), recall.result().numpy(), accuracy.result().numpy()

def train(epochs=10, batch_size=1024, lr=0.001, dim=128, eva_per_epochs=1):
    # 读取数据
    train_infos, test_infos, all_items = dp.getTrainAndTestSeqs(fp.Ml_latest_small.SEQS)
    # 初始化模型
    model = RNNALSRec(max(all_items) + 1, dim)

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=5e-3)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_infos).batch(batch_size).shuffle(10000)

    for e in range(epochs):
        e_loss = tf.keras.metrics.Mean()
        for seq_batch in train_dataset:
            x = seq_batch[:, :-2]
            item = seq_batch[:, -2]
            y = seq_batch[:, -1]

            with tf.GradientTape() as tape:
                pred = model(x, item)
                loss = loss_fn(y, pred)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            e_loss.update_state(loss)

        print(f"Epoch {e} | Loss: {e_loss.result():.4f}")

        if e % eva_per_epochs == 0:
            p, r, acc = doEva(model, train_infos)
            print(f"Train   p: {p:.4f} | r: {r:.4f} | acc: {acc:.4f}")
            p, r, acc = doEva(model, test_infos)
            print(f"Test    p: {p:.4f} | r: {r:.4f} | acc: {acc:.4f}")

def main():
    train()


if __name__ == "__main__":
    exit(main())
