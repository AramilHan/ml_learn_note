# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/3/7 21:31
@File    : LR_tensorflow.py
@Description : Tensorflow实现逻辑回归（LR）
"""
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import dataloader4ml100kOneHot


class LR(tf.keras.Model):
    def __init__(self, n_features):
        """
        :param n_features: 特征数量
        """
        super(LR, self).__init__()
        initializer = tf.keras.initializers.GlorotUniform()
        self.w = self.add_weight(initializer=initializer,
                                 shape=(n_features, 1),
                                 trainable=True,
                                 name="weights")
        self.b = self.add_weight(initializer=initializer,
                                 shape=(1,),
                                 trainable=True,
                                 name="bias")

    def call(self, x):
        logits = tf.matmul(x, self.w) + self.b
        return tf.sigmoid(logits)

def doEva(net, x, y):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    preds = net(x, training=False)
    y_pred = (preds.numpy() >= 0.5).astype(int).flatten()
    y_true = y.numpy().astype(int).flatten()
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc

def train(epochs=20, batch_size=1024, lr=0.01, eva_per_epochs=1):
    # 读取数据
    x_train, x_test, y_train, y_test = dataloader4ml100kOneHot.read_data()
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    # 转换为TensorFlow Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).shuffle(buffer_size=10000).batch(batch_size)
    # 获取特征维度
    features = x_train.shape[1]
    # 初始化模型
    model = LR(features)
    # 定义优化器和损失函数
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    # 训练循环
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                preds = model(x_batch, training=True)
                loss = loss_fn(y_batch, preds)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss.numpy()
        avg_loss = total_loss / (len(x_train) // batch_size)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        # 评估模型
        if epoch % eva_per_epochs == 0:
            p, r, acc = doEva(model, x_train, y_train)
            print(f"Train  Precision: {p:.4f} | Recall: {r:.4f} | Accuracy: {acc:.4f}")
            p, r, acc = doEva(model, x_test, y_test)
            print(f"Test  Precision: {p:.4f} | Recall: {r:.4f} | Accuracy: {acc:.4f}")

def main():
    train()


if __name__ == "__main__":
    exit(main())
