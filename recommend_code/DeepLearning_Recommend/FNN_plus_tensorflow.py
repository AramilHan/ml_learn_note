# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/4/6 15:47
@File    : FNN_plus_tensorflow.py
@Description : 端到端训练FNN
"""
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, Model, constraints, Sequential

sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../'))
from Logistic_Regression import dataloader4ml100kIndexs

class FNNPlus(Model):
    def __init__(self, n_features, user_df, item_df, dim=128):
        super(FNNPlus, self).__init__()
        # 随机初始化所有特征的特征向量
        self.features = layers.Embedding(n_features, dim, embeddings_constraint=constraints.MaxNorm(1))
        self.mlp_layer = self.__mlp(dim)
        # 记录好用户和物品的特征索引
        self.user_df = tf.convert_to_tensor(user_df.values)
        self.item_df = tf.convert_to_tensor(item_df.values)

    def __mlp(self, dim):
        return Sequential([
            layers.Dense(dim // 2, activation='tanh'),
            layers.Dense(dim // 4, activation='tanh'),
            layers.Dense(1, activation='sigmoid')
        ])

    # FM聚合层
    def aggregator_FM(self, feature_embs):
        # feature_embs: [batch_size, n_features, dim]
        # [batch_size, dim]
        square_of_sum = tf.reduce_sum(feature_embs, axis=1) ** 2
        # [batch_size, dim]
        sum_of_square = tf.reduce_sum(feature_embs ** 2, axis=1)
        # [batch_size, dim]
        agg_output = square_of_sum - sum_of_square
        return agg_output

    # 把用户和物品的特征拼接起来
    def __get_all_features(self, u, i):
        user_ids = tf.gather(self.user_df, u - 1)
        item_ids = tf.gather(self.item_df, i - 1)
        all_features = tf.concat([user_ids, item_ids], axis=1)
        return all_features

    def call(self, inputs):
        u, i = inputs['u'], inputs['i']
        all_feature_index = self.__get_all_features(u, i)
        all_feature_embs = self.features(all_feature_index)
        out = self.aggregator_FM(all_feature_embs)
        out = self.mlp_layer(out)
        out = tf.squeeze(out)
        return out

def do_eva(model, test_triples):
    d = tf.convert_to_tensor(test_triples, dtype=tf.float32)
    u = tf.cast(d[:, 0], dtype=tf.int32)
    i = tf.cast(d[:, 1], dtype=tf.int32)
    r = d[:, 2]
    out = model({'u': u, 'i': i})
    y_pred = tf.where(out >= 0.5, 1., 0.)
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    accuracy = tf.keras.metrics.Accuracy()

    precision.update_state(y_pred, r)
    recall.update_state(y_pred, r)
    accuracy.update_state(y_pred, r)
    return precision.result().numpy(), recall.result().numpy(), accuracy.result().numpy()

def train(epochs=20, batch_size=1024, lr=0.02, dim=128, eva_per_epochs=1, need_eva=True):
    train_triples, test_triples, user_df, item_df, n_features = dataloader4ml100kIndexs.read_data()
    train_dataset = tf.data.Dataset.from_tensor_slices(train_triples).batch(batch_size).shuffle(buffer_size=10000)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.3)
    model = FNNPlus(n_features, user_df, item_df, dim)
    for e in range(epochs):
        all_loss = tf.keras.metrics.Mean()

        for triple_batch in train_dataset:
            u = tf.cast(triple_batch[:, 0], dtype=tf.int32)
            i = tf.cast(triple_batch[:, 1], dtype=tf.int32)
            r = triple_batch[:, 2]
            with tf.GradientTape() as tape:
                pred = model({'u': u, 'i': i})
                loss = loss_fn(r, pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            all_loss.update_state(loss)
        print(f"Epoch {e}, Loss: {all_loss.result():.4f}")
        if e % eva_per_epochs == 0:
            p, r, acc = do_eva(model, train_triples)
            print(f"Train p: {p:.4f} | r: {r:.4f} | acc: {acc:.4f}")
            p, r, acc = do_eva(model, test_triples)
            print(f"Test p: {p:.4f} | r: {r:.4f} | acc: {acc:.4f}")

def main():
    train()


if __name__ == "__main__":
    exit(main())
