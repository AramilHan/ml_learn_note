# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/3/8 15:48
@File    : POLY2_tensorflow.py
@Description : 
"""
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sympy.physics.vector import gradient

import dataloader4ml100kOneHot

class POLY2(tf.keras.Model):
    def __init__(self, n_features):
        super(POLY2, self).__init__()
        # 初始化参数
        initializer = tf.keras.initializers.GlorotUniform()
        self.w0 = self.add_weight(
            name="w0",
            shape=(1, ),
            initializer=initializer,
            trainable=True)
        self.w1 = self.add_weight(
            name="w1",
            shape=(n_features, 1),
            initializer=initializer,
            trainable=True)
        self.w2 = self.add_weight(
            name="w2",
            shape=(n_features, n_features),
            initializer=initializer,
            trainable=True)

    def crossLayer(self, x):
        # 生成交叉项 [batch_size, n_feats, n_feats]
        x_left = tf.expand_dims(x, axis=2)  # [batch_size, n_feats, 1]
        x_right = tf.expand_dims(x, axis=1)  # [batch_size, 1, n_feats]
        x_cross = tf.matmul(x_left, x_right)  # [batch_size, n_feats, n_feats]
        # 计算交叉项输出 [batch_size]
        cross_out = tf.reduce_sum(tf.reduce_sum(x_cross * self.w2, axis=2),
                                  axis=1,
                                  keepdims=True)
        return cross_out

    def call(self, x):
        # 线性部分 [batch, 1]
        lr_out = self.w0 + tf.matmul(x, self.w1)
        # 交叉项部分 [batch, 1]
        cross_out = self.crossLayer(x)
        # 合并结果 [batch, 1]
        logits = tf.sigmoid(lr_out + cross_out)
        return logits

def doEva(net, x, y):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    preds = net(x, training=False).numpy().flatten()
    y_pred = (preds >= 0.5).astype(int)
    y_true = y.numpy().astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc

def train(epochs=20, batch_size=1024, lr=0.01, eva_per_epochs=1):
    # 读取并转换数据
    x_train, x_test, y_train, y_test = dataloader4ml100kOneHot.read_data()
    # 转换为numpy数组
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # 创建 TensorFlow DataSet
    train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                     .shuffle(buffer_size=10000)
                     .batch(batch_size)
                     .prefetch(tf.data.AUTOTUNE))

    # 初始化模型
    features = x_train.shape[1]
    model = POLY2(features)

    # 定义优化器和损失函数
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # 训练循环
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                preds = model(x_batch, training=True)
                # 添加维度已匹配损失函数的要求 [batch, 1] -> [batch]
                loss = loss_fn(y_batch, preds)

            gradients = tape.gradient(loss, model.trainable_variables)
            # 梯度裁剪防止爆炸
            gradients = [tf.clip_by_norm(g, 5.0) for g in gradients]
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss.numpy()

        avg_loss = total_loss / (len(x_train) // batch_size)
        print(f"Epoch {epoch}, loss: {avg_loss:.4f}")

        # 模型评估
        if epoch % eva_per_epochs == 0:
            p, r, acc = doEva(model, x_train, y_train)
            print(f"Train -> p: {p:.4f} | r: {r:.4f} | acc: {acc:.4f}")
            p, r, acc = doEva(model, x_test, y_test)
            print(f"Test -> p: {p:.4f} | r: {r} | acc: {acc:.4f}")

def main():
    train()


if __name__ == "__main__":
    exit(main())
