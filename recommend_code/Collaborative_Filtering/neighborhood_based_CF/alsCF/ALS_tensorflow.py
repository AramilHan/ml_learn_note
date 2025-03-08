# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/3/2 22:07
@File    : ALS_tensorflow.py
@Description : 
"""
import os
import sys
import numpy as np
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
from data_set import filepaths as fp
from Collaborative_Filtering import dataloader
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, accuracy_score


class ALS(tf.keras.Model):
    def __init__(self, n_users, n_items, dim):
        super(ALS, self).__init__()
        self.n_users = tf.keras.layers.Embedding(n_users, dim,
                                                 embeddings_constraint=tf.keras.constraints.UnitNorm(axis=1))
        self.n_items = tf.keras.layers.Embedding(n_items, dim,
                                                 embeddings_constraint=tf.keras.constraints.UnitNorm(axis=1))

    def call(self, u, v):
        u_vec = self.n_users(u)
        v_vec = self.n_items(v)
        uv = tf.reduce_sum(u_vec * v_vec, axis=1)
        logit = tf.sigmoid(uv)
        return logit


def doEva(net, d):
    u = tf.convert_to_tensor(d[:, 0], dtype=tf.int32)
    v = tf.convert_to_tensor(d[:, 1], dtype=tf.int32)
    r = d[:, 2]
    out = net(u, v, training=False).numpy()
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = np.where(r == 1, 1, 0)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc


def train(epochs=10, batch_size=1024, lr=0.01, dim=64, eva_per_epochs=1):
    user_set, item_set, train_set, test_set = dataloader.readRecData(fp.Ml_100K.RATING5, test_ratio=0.1)
    net = ALS(len(user_set), len(item_set), dim)

    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)
    criterion = tf.keras.losses.BinaryCrossentropy()
    train_set = np.array(train_set, dtype=np.float32)
    test_set = np.array(test_set, dtype=np.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_set[:, 0].astype(np.int32),
        train_set[:, 1].astype(np.int32),
        train_set[:, 2].astype(np.float32)
    )).shuffle(len(train_set)).batch(batch_size)
    for epoch in range(epochs):
        all_loss = 0
        for batch in train_dataset:
            u, i, r = batch
            r = tf.where(r  == 1, 1.0, 0.0)
            with tf.GradientTape() as tape:
                result = net(u, i, training=True)
                loss = criterion(r, result)

            gradients = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(zip(gradients, net.trainable_variables))
            all_loss += loss.numpy()

        avg_loss = all_loss / (len(train_set) // batch_size)
        print(f"epoch: {epoch}, avg_loss: {avg_loss:.4f}")

        if epoch % eva_per_epochs == 0:
            p, r, acc = doEva(net, train_set)
            print(f"train: Precision: {p:.4f} | Recall: {r:.4f} | Accuracy: {acc:.4f}")
            p, r, acc = doEva(net, test_set)
            print(f"test: Precision: {p:.4f} | Recall: {r:.4f} | Accuracy: {acc:.4f}")


def main():
    train()


if __name__ == "__main__":
    exit(main())
