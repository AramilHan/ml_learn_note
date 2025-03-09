# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/3/8 23:03
@File    : FM_tensorflow.py
@Description : 
"""
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import dataloader4ml100kOneHot

class FM(tf.keras.Model):
    def __init__(self, n_features, dim):
        super(FM, self).__init__()
        initializer = tf.keras.initializers.GlorotUniform()
        self.w0 = self.add_weight(name="w0",
                                  initializer=initializer,
                                  shape=(1,),
                                  trainable=True)
        self.w1 = self.add_weight(name="w1",
                                  initializer=initializer,
                                  shape=(n_features, 1),
                                  trainable=True)
        self.w2 = self.add_weight(name="w2",
                                  initializer=initializer,
                                  shape=(n_features, dim),
                                  trainable=True)

    def FM_cross(self, x):
        # [batch_size, dim]
        square_of_sum = tf.matmul(x, self.w2) ** 2
        # [batch_size, dim]
        sum_of_square = tf.matmul(x ** 2, self.w2 ** 2)

        output = square_of_sum - sum_of_square
        output = tf.reduce_sum(output, axis=1, keepdims=True)
        output = 0.5 * output
        return output

    def call(self, x):
        lr_out = self.w0 + tf.matmul(x, self.w1)
        cross_out = self.FM_cross(x)
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

def train(epochs=20, batch_size=1024, lr=0.01, dim=128, eva_per_epochs=1):
    # 读取数据
    x_train, x_test, y_train, y_test = dataloader4ml100kOneHot.read_data()
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # 创建 TensorFlow DataSet
    train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                     .shuffle(buffer_size=10000)
                     .batch(batch_size)
                     .prefetch(tf.data.AUTOTUNE))
    # 初始化模型
    features = x_train.shape[1]
    model = FM(features, dim)

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
            gradients = [tf.clip_by_norm(g, 0.5) for g in gradients]
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss.numpy()

        avg_loss = total_loss / (len(x_train) // batch_size)
        print(f"Epoch: {epoch}, loss: {avg_loss:.4f}")

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
