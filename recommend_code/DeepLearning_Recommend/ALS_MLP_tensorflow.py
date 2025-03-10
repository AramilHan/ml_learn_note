# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/3/9 17:06
@File    : ALS_MLP_tensorflow.py
@Description : 
"""
import os
import sys
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import precision_score, recall_score, accuracy_score
from Collaborative_Filtering import dataloader
from data_set import filepaths as fp

class ALS_MLP(keras.Model):
    def __init__(self, n_user, n_items, dim):
        super(ALS_MLP, self).__init__()
        # 用户嵌入层，添加L2范数约束
        self.user_embedding = keras.layers.Embedding(n_user,
                                                     dim,
                                                     embeddings_constraint=keras.constraints.UnitNorm(axis=1))
        # 物品嵌入层，添加L2范数约束
        self.item_embedding = keras.layers.Embedding(n_items,
                                                     dim,
                                                     embeddings_constraint=keras.constraints.UnitNorm(axis=1))

        # 用户MLP
        self.user_mlp = keras.Sequential([
            keras.layers.Dense(dim//2, activation="tanh"),
            keras.layers.Dense(dim//4, activation="tanh"),
            keras.layers.Dropout(0.5)
        ])
        # 物品MLP
        self.item_mlp = keras.Sequential([
            keras.layers.Dense(dim//2, activation="tanh"),
            keras.layers.Dense(dim//4, activation="tanh"),
            keras.layers.Dropout(0.5)
        ])

    def call(self, inputs, training=None):
        u_ids, i_ids = inputs
        # 获取嵌入向量 [batch_size, dim]
        u_embedding = self.user_embedding(u_ids)
        i_embedding = self.item_embedding(i_ids)

        # 通过MLP [batch_size, dim//4]
        u_mlp = self.user_mlp(u_embedding, training=training)
        i_mlp = self.item_mlp(i_embedding, training=training)

        # 点击计算 [batch_size]
        logit = tf.reduce_sum(u_mlp * i_mlp, axis=1)
        return tf.sigmoid(logit * 3)

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
    loss_fn = keras.losses.BinaryCrossentropy()
    # 定义优化器
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    # 训练循环
    for epoch in range(epochs):
        epoch_loss = 0
        for step, ((u_batch, i_batch), r_batch) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                preds = model.call((u_batch, i_batch), training=True)
                loss = loss_fn(r_batch, preds)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss.numpy()
        print(f"Epoch {epoch} Loss: {epoch_loss/(step + 1):.4f}")

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
