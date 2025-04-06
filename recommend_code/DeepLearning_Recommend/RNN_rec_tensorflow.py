# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/3/10 23:43
@File    : RNN_rec_tensorflow.py
@Description : 
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
import RNN_data_prepare as dp
from data_set import filepaths as fp

class RNNRec(Model):
    def __init__(self, n_items, hidden_size=64, dim=128):
        super(RNNRec, self).__init__()
        self.item_embeddings = layers.Embedding(n_items, hidden_size, embeddings_constraint=tf.keras.constraints.MaxNorm(1))
        self.rnn = layers.SimpleRNN(hidden_size, return_sequences=False, return_state=False)
        self.dense1 = self.dense_layer(hidden_size // 2)
        self.dense2 = self.dense_layer(1)

    def dense_layer(self, out_features):
        return tf.keras.Sequential([
            layers.Dense(out_features, activation='tanh')
        ])

    def call(self, inputs, training=None):
        item_embedding = self.item_embeddings(inputs)
        hn = self.rnn(item_embedding)
        if training:
            hn = tf.nn.dropout(hn, rate=0.5)
        output = self.dense1(hn)
        output = self.dense2(output)
        output = tf.squeeze(output, axis=-1)
        return tf.sigmoid(output)

def doEva(model, test_triples):
    x = test_triples[:, :-1]
    y = test_triples[:, -1]

    out = model(x, training=False)
    y_pred = tf.where(out >= 0.5, 1., 0.)

    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    accuracy = tf.keras.metrics.Accuracy()

    precision.update_state(y, y_pred)
    recall.update_state(y, y_pred)
    accuracy.update_state(y, y_pred)
    return precision.result().numpy(), recall.result().numpy(), accuracy.result().numpy()

def train(epochs=10, batch_size=1024, lr=0.001, rnn_hidden_size=64, dim=128, eva_per_epochs=1):
    # 读取数据
    train_infos, test_infos, all_items = dp.getTrainAndTestSeqs(fp.Ml_latest_small.SEQS)
    # 初始化模型
    model = RNNRec(max(all_items) + 1, rnn_hidden_size, dim)

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=5e-3)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_infos).batch(batch_size).shuffle(10000)
    for e in range(epochs):
        e_loss = tf.keras.metrics.Mean()
        for seq_batch in train_dataset:
            x = seq_batch[:, :-1]
            y = seq_batch[:, -1]

            with tf.GradientTape() as tape:
                pred = model(x, training=True)
                loss = loss_fn(y, pred)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            e_loss.update_state(loss)

        print(f"Epoch {e} | Loss: {e_loss.result():.4f}")

        if e % eva_per_epochs == 0:
            p, r, acc = doEva(model, train_infos)
            print(f"Train: p: {p:.4f}, r: {r:.4f}, acc: {acc:.4f}")
            p, r, acc = doEva(model, test_infos)
            print(f"Test: p: {p:.4f}, r: {r:.4f}, acc: {acc:.4f}")

def main():
    train()


if __name__ == "__main__":
    exit(main())
