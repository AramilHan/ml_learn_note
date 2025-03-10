# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/3/9 22:10
@File    : ALS_MLP_concat_tensorflow.py
@Description : 
"""
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, accuracy_score
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
from Collaborative_Filtering import dataloader
from data_set import filepaths as fp

class ALS_MLP(tf.keras.Model):
    def __init__(self, n_users, n_items, dim):
        super(ALS_MLP, self).__init__()
        # 用户embedding
        self.user_embedding = tf.keras.layers.Embedding(n_users, dim,
                                                        embeddings_constraint=tf.keras.constraints.UnitNorm(axis=1))
        # 物品embedding
        self.item_embedding = tf.keras.layers.Embedding(n_items, dim,
                                                        embeddings_constraint=tf.keras.constraints.UnitNorm(axis=1))

        # 拼接后的MLP
        self.concat_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(dim, actication="tanh"),
            tf.keras.layers.Dense(dim // 2, actication="tanh"),
            tf.keras.layers.Dense(1, actication="tanh"),
            tf.keras.layers.Dropout(0.2)
        ])

    def call(self, inputs, training=None):
        u_ids, i_ids = inputs
        # 获取嵌入向量 [batch_size, dim]
        u_embedding = self.user_embedding(u_ids)
        i_embedding = self.item_embedding(i_ids)

        # 拼接用户物品向量 [batch_size, dim * 2]
        uv_embedding = tf.concat([u_embedding, i_embedding], axis=1)

        # 通过MLP
        concat_mlp = self.concat_mlp(uv_embedding, training=training)
        return tf.sigmoid(concat_mlp)

def doEva(model, data):
    data = np.array(data)
    u, i, r = (data[:, 0].astype(np.int32),
               data[:, 1].astype(np.int32),
               data[:, 2].astype(np.float32))

    preds = model.predict((u, i), verbose=0).flatten()
    y_pred = (preds >= 0.5).astype(int)
    y_true = (r >= 0.5).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, accuracy


def train(epochs=10, batch_size=1024, lr=0.001, dim=256, eva_per_epochs=1):
    # 读取数据
    user_set, item_set, train_set, test_set = dataloader.readRecData(fp.Ml_100K.RATING, test_ratio=0.1)

    # 转换为TensorFlow DataSet
    def gen_train():
        for u, i, r in train_set:
            yield (u, i), r

    train_ds = tf.data.Dataset.from_generator(
        gen_train,
        output_signature=(
            (tf.TensorSpec(shape=(), dtype=tf.int32),
             tf.TensorSpec(shape=(), dtype=tf.int32)),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )).shuffle(10000).batch(batch_size).prefetch(2)

    # 初始化模型
    model = ALS_MLP(len(user_set), len(item_set), dim)
    # 定义损失函数
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    # 定义优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # 训练循环
    for e in range(epochs):
        epoch_loss = 0
        for step, ((u_batch, i_batch), r_batch) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                preds = model((u_batch, i_batch), training=True)
                loss = loss_fn(r_batch, preds)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss.numpy()
        print(f"Epoch: {e}, Loss: {epoch_loss / (step + 1):.4f}")

        # 评估模型
        if epoch % eva_per_epochs == 0:
            p, r, acc = doEva(model, train_set)
            print(f"Train p: {p:.4f} | r: {r:.4f} | acc: {acc:.4f}")
            p, r, acc = doEva(model, test_set)
            print(f"Test p: {p:.4f} | r: {r:.4f} | acc: {acc:.4f}")


def main():
    train()


if __name__ == "__main__":
    exit(main())
