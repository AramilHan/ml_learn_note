# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/4/8 13:58
@File    : AFM_tensorflow.py
@Description : 
"""
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
from Logistic_Regression import dataloader4ml100kIndexs

class AFM(Model):
    def __init__(self, n_features, user_df, item_df, k, t):
        super(AFM, self).__init__()
        self.features = layers.Embedding(n_features, k, embeddings_constraint=tf.keras.constraints.max_norm(1.))
        self.attention_linear = layers.Dense(t, activation='relu')
        self.h = self.add_weight(shape=(t, 1), initializer='glorot_uniform', trainable=True)
        self.p = self.add_weight(shape=(k, 1), initializer='glorot_uniform', trainable=True)
        self.user_df = user_df
        self.item_df = item_df

    def fm_aggregator(self, feature_embs):
        square_of_sum = tf.square(tf.reduce_sum(feature_embs, axis=1))
        sum_of_square = tf.reduce_sum(tf.square(feature_embs), axis=1)
        output = square_of_sum - sum_of_square
        return output

    def attention(self, embs):
        embs = self.attention_linear(embs)
        embs = tf.nn.relu(embs)
        embs = tf.matmul(embs, self.h)
        atts = tf.nn.softmax(embs, axis=1)
        return atts

    def get_all_features(self, u, i):
        users = tf.convert_to_tensor(self.user_df.loc[u.numpy()].values, dtype=tf.int32)
        items = tf.convert_to_tensor(self.item_df.loc[i.numpy()].values, dtype=tf.int32)
        all_features = tf.concat([users, items], axis=1)
        return all_features

    def call(self, inputs):
        u, i = inputs
        all_feature_index = self.get_all_features(u, i)
        all_feature_embs = self.features(all_feature_index)
        embs = self.fm_aggregator(all_feature_embs)
        atts = self.attention(embs)
        output = tf.matmul(atts * embs, self.p)
        output = tf.squeeze(output)
        logit = tf.nn.sigmoid(output)
        return logit


def do_eva(model, test_triples):
    d = tf.convert_to_tensor(test_triples, dtype=tf.float32)
    u = tf.cast(d[:, 0], dtype=tf.int32)
    i = tf.cast(d[:, 1], dtype=tf.int32)
    r = d[:, 2]
    out = model((u, i))
    y_pred = tf.where(out >= 0.5, 1., 0.)
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    accuracy = tf.keras.metrics.Accuracy()
    precision.update_state(y_pred, r)
    recall.update_state(y_pred, r)
    accuracy.update_state(y_pred, r)
    return precision.result().numpy(), recall.result().numpy(), accuracy.result().numpy()

def train(epochs=20, batch_size=1024, lr=0.02, k=128, t=64, eva_per_epochs=1, need_eva=True):
    train_triples, test_triples, user_df, item_df, n_features = dataloader4ml100kIndexs.read_data()
    model = AFM(n_features, user_df, item_df, k, t)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.3)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_triples).batch(batch_size).shuffle(buffer_size=10000)

    for e in range(epochs):
        all_loss = 0
        for batch in train_dataset:
            u = tf.cast(batch[:, 0], dtype=tf.int32)
            i = tf.cast(batch[:, 1], dtype=tf.int32)
            r = batch[:, 2]
            with tf.GradientTape() as tape:
                r = tf.cast(r, dtype=tf.float32)
                logit = model((u, i))
                loss = loss_fn(y_true=r, y_pred=logit)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                all_loss += loss

        print(f"epoch {e}  avg_loss: {all_loss / (len(train_triples) // batch_size): .4f}")

        if e % eva_per_epochs == 0 and need_eva:
            p, r, acc = do_eva(model, train_triples)
            print(f"Train p: {p:.4f}, r: {r:.4f}, acc: {acc:.4f}")
            p, r, acc = do_eva(model, test_triples)
            print(f"Test p: {p:.4f}, r: {r:.4f}, acc: {acc:.4f}")


def main():
    train()


if __name__ == "__main__":
    exit(main())
