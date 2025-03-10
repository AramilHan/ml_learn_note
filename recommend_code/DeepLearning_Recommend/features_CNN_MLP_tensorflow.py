# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/3/10 17:14
@File    : features_CNN_MLP_tensorflow.py
@Description : 
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, constraints, Sequential
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
from Logistic_Regression import dataloader4ml100kIndexs
from data_set import filepaths as fp

class FeaturesCNN(Model):
    def __init__(self, n_user_features, n_item_features, user_df, item_df, dim, **kwargs):
        super(FeaturesCNN, self).__init__(**kwargs)
        self.user_embeddings = layers.Embedding(n_user_features, dim, embeddings_constraint=constraints.MaxNorm(1))
        self.item_embeddings = layers.Embedding(n_item_features, dim, embeddings_constraint=constraints.MaxNorm(1))

        self.user_df = tf.convert_to_tensor(user_df.values)
        self.item_df = tf.convert_to_tensor(item_df.values)

        total_features_num = user_df.shape[1] + item_df.shape[1]

        self.conv = layers.Conv1D(filters=1, kernel_size=3, input_shape=(total_features_num, dim))

        self.dense1 = self.dense_layer(dim // 2)
        self.dense2 = self.dense_layer(1)
        self.sigmoid = tf.keras.activations.sigmoid

    def dense_layer(self, out_features):
        return Sequential([
            layers.Dense(out_features, activation='tanh')
        ])

    def call(self, inputs, training=None):
        u, i = inputs['u'], inputs['i']
        try:
            user_ids = tf.gather(self.user_df, u-1)
            item_ids = tf.gather(self.item_df, i-1)
        except Exception as e:
            print(e)

        user_embeddings = self.user_embeddings(user_ids)
        item_embeddings = self.item_embeddings(item_ids)

        uv = tf.concat([user_embeddings, item_embeddings], axis=1)
        uv = tf.transpose(uv, perm=[0, 2, 1])
        uv = self.conv(uv)
        uv = tf.squeeze(uv)

        uv = self.dense1(uv)
        if training:
            uv = tf.nn.dropout(uv, rate=0.5)
        uv = self.dense2(uv)
        uv = tf.squeeze(uv)
        return tf.sigmoid(uv)

def doEva(model, test_triples):
    d = tf.convert_to_tensor(test_triples, dtype=tf.float32)
    u = tf.cast(d[:, 0], dtype=tf.int32)
    i = tf.cast(d[:, 1], dtype=tf.int32)
    r = d[:, 2]
    out = model({'u': u, 'i': i}, training=False)
    y_pred = tf.where(out >= 0.5, 1., 0.)
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    accuracy = tf.keras.metrics.Accuracy()

    precision.update_state(r, y_pred)
    recall.update_state(r, y_pred)
    accuracy.update_state(r, y_pred)
    return precision.result().numpy(), recall.result().numpy(), accuracy.result().numpy()

def train(epochs=10, batch_size=1024, lr=0.001, dim=256, eva_per_epochs=1):
    # 读取数据
    train_triples, test_triples, user_df, item_df, n_user_features, n_item_features = dataloader4ml100kIndexs.read_data_user_item_df()
    # 构建数据管道
    train_dataset = tf.data.Dataset.from_tensor_slices(train_triples).batch(batch_size).shuffle(buffer_size=10000)
    # 定义损失函数
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    # 定义优化器
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=5e-3)
    # 定义模型
    model = FeaturesCNN(n_user_features, n_item_features, user_df, item_df, dim)
    for e in range(epochs):
        e_loss = tf.keras.metrics.Mean()

        for triple_batch in train_dataset:
            u = tf.cast(triple_batch[:, 0], dtype=tf.int32)
            i = tf.cast(triple_batch[:, 1], dtype=tf.int32)
            r = triple_batch[:, 2]
            with tf.GradientTape() as tape:
                pred = model({'u': u, 'i': i}, training=True)
                loss = loss_fn(r, pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            e_loss.update_state(loss)
        print(f"Epoch {e}, Loss: {e_loss.result():.4f}")
        if e % eva_per_epochs == 0:
            p, r, acc = doEva(model, train_triples)
            print(f"Train p: {p:.4f} | r: {r:.4f} | acc: {acc:.4f}")
            p, r, acc = doEva(model, test_triples)
            print(f"Test p: {p:.4f} | r: {r:.4f} | acc: {acc:.4f}")

def main():
    train()


if __name__ == "__main__":
    exit(main())
