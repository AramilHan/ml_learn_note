# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/3/9 23:09
@File    : features_MLP_tensorflow.py
@Description : 
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, metrics
import pandas as pd
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
from Logistic_Regression import dataloader4ml100kIndexs
from data_set import filepaths as fp

class features_MLP(tf.keras.Model):
    def __init__(self, n_user_features, n_item_features, user_df, item_df, dim):
        super(features_MLP, self).__init__()
        # 设置用户特征embedding
        self.user_embeddings = layers.Embedding(n_user_features, dim, embeddings_constraint=tf.keras.constraints.max_norm(1))
        # 设置物品特征embedding
        self.item_embeddings = layers.Embedding(n_item_features, dim, embeddings_constraint=tf.keras.constraints.max_norm(1))

        # 记录用户和物品的特征索引
        self.user_df = user_df
        self.item_df = item_df

        # 得到用户和物品特征种类数量之和
        total_features_num = user_df.shape[1] + item_df.shape[1]

        # 定义MLP
        self.mlp_dense = tf.keras.Sequential([
            layers.Dense(dim * total_features_num // 2, activation="tanh"),
            layers.Dense(dim, activation="tanh"),
            layers.Dense(1, activation="tanh"),
            layers.Dropout(0.2)
        ])

    def call(self, u, i, training=None):
        user_ids = tf.convert_to_tensor(self.user_df.loc[u.numpy()].values)
        item_ids = tf.convert_to_tensor(self.item_df.loc[i.numpy()].values)

        user_features = self.user_embeddings(user_ids)
        item_features = self.item_embeddings(item_ids)

        uv = tf.concat([user_features, item_features], axis=1)
        uv = tf.reshape(uv, (uv.shape[0], -1))

        uv = self.mlp_dense(uv)
        uv = tf.squeeze(uv)
        return tf.sigmoid(uv)


def doEva(model, test_triples):
    u, i, r = zip(*test_triples)
    u = tf.convert_to_tensor(u)
    i = tf.convert_to_tensor(i)
    with tf.GradientTape() as tape:
        out = model(u, i)
    y_pred = [1 if i >= 0.5 else 0 for i in out]
    precision_metric = metrics.Precision()
    recall_metric = metrics.Recall()
    accuracy_metric = metrics.Accuracy()

    precision_metric.update_state(r, y_pred)
    recall_metric.update_state(r, y_pred)
    accuracy_metric.update_state(r, y_pred)

    precision = precision_metric.result().numpy()
    recall = recall_metric.result().numpy()
    accuracy = accuracy_metric.result().numpy()
    return precision, recall, accuracy

def train(epochs=10, batch_size=1024, lr=0.001, dim=256, eva_per_epochs=1):
    train_triples, test_triples, user_df, item_df, n_user_features, n_item_features = dataloader4ml100kIndexs.read_data_user_item_df()
    # 初始化模型
    model = features_MLP(n_user_features, n_item_features, user_df, item_df, dim)
    # 定义损失函数
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    # 定义优化器
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=5e-3)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_triples).shuffle(buffer_size=1024).batch(batch_size)

    for e in range(epochs):
        all_loss = 0
        for batch in train_dataset:
            u, i, r = batch[:, 0], batch[:, 1], batch[:, 2]
            with tf.GradientTape() as tape:
                logit = model(u, i)
                loss = loss_fn(r, logit)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            all_loss += loss
        avg_loss = all_loss / (len(train_triples) // batch_size)
        print(f"epoch {e}, loss: {avg_loss:.4f}")

        if e % eva_per_epochs == 0:
            precision, recall, accuracy = doEva(model, train_triples)
            print(f"Train  p: {precision:.4f}, r: {recall:.4f}, accuracy: {accuracy:.4f}")
            precision, recall, accuracy = doEva(model, test_triples)
            print(f"Test  p: {precision:.4f}, r: {recall:.4f}, accuracy: {accuracy:.4f}")


def main():
    train()


if __name__ == "__main__":
    exit(main())
