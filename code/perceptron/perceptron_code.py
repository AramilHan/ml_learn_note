# -*- encoding: utf-8 -*-
"""
    感知机模型实现方式
"""
import numpy as np

def perceptron_python_demo(n):
    w = np.array([0, 0])
    b = 0
    train_set = [
        [np.array([3, 3]), 1],
        [np.array([4, 3]), 1],
        [np.array([1, 1]), -1]
    ]
    def perceptron(x, y, w, b):
        dot = np.dot(x, w)
        l = y * (dot + b)
        if l > 0:
            return w, b
        w1 = w + np.array(y)*np.array(x)
        b1 = b + y
        return w1, b1
    for i in range(n):
        for x, y in train_set:
            w, b = perceptron(x, y, w, b)
            print(w, b)
    print(f"result: {w}, {b}")

def perceptron_duality_python_demo():
    X = np.array([[3, 3], [4, 3], [1, 1]])
    Y = np.array([1, 1, -1])
    N, d = X.shape
    alpha = np.zeros(N)
    b = 0
    eta = 1
    Garm = np.dot(X, X.T)
    while True:
        has_misclassified = False
        for i in range(N):
            pred = np.sum(alpha * Y * Garm[i]) + b
            # 检查是否误分类
            if pred * Y[i] <= 0:
                alpha[i] += 1
                b += Y[i]
                has_misclassified = True
        if not has_misclassified:
            break
    print(alpha, b)
    return alpha, b

def perceptron_scikit_learn_demo():
    from sklearn.linear_model import Perceptron
    import numpy as np
    X = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])
    # 初始化感知机模型
    clf = Perceptron(max_iter=10000, tol=1e-3, random_state=0)
    # 训练模型
    clf.fit(X, y)
    print(f"权重向量w: {clf.coef_}")
    print(f"偏置b: {clf.intercept_}")

def perceptron_tensorflow_demo():
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.optimizers import SGD
    x_train = np.array([[3, 3], [4, 3], [1, 1]])
    y_train = np.array([1, 1, -1])
    y_train = (y_train + 1) / 2
    # 构建感知机模型
    model = Sequential([Input(shape=(2,)), Dense(1, activation='sigmoid')])
    # 编译模型
    model.compile(optimizer=SGD(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])
    # 训练模型
    model.fit(x_train, y_train, epochs=100, verbose=1)
    # 测试模型
    x_test = np.array([[2, 2], [5, 5], [1, 1]])
    y_pred = model.predict(x_test)
    print(y_pred)
    y_pred_classes = (y_pred > 0.5).astype(int) * 2 -1
    print(f"测试输出: {y_pred_classes.flatten()}")

def perceptron_pytorch_demo():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    x_train = torch.tensor([[3.0, 3.0], [4.0, 3.0], [1.0, 1.0]])
    y_train = torch.tensor([1.0, 1.0, -1.0])
    y_train = (y_train + 1) / 2
    # 定义感知机模型
    class Perceptron(nn.Module):
        def __init__(self):
            super(Perceptron, self).__init__()
            self.fc = nn.Linear(2, 1)  # 输入维度为2，输出维度为1

        def forward(self, x):
            # 前向传播
            return self.fc(x)

    # 实例化模型
    model = Perceptron()
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=0.1)  # 随机梯度下降优化器

    # 训练模型
    epochs = 1000
    for epoch in range(epochs):
        # 前向传播
        outputs = model(x_train).squeeze()
        loss = criterion(outputs, y_train)

        # 反向传播和优化
        optimizer.zero_grad()  # 清空维度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        print(f"Epoch: [{epoch + 1}/{epochs}], loss: {loss.item():.4f}")

    # 测试模型
    x_test = torch.tensor([[2.0, 2.0], [5.0, 5.0], [1.0, 1.0]])
    with torch.no_grad():  # 不计算梯度
        y_pred = model(x_test).squeeze()
        y_pred_classes = (torch.sigmoid(y_pred) > 0.5).float()
        y_pred_classes = y_pred_classes * 2 - 1
    print(f"测试样本预测结果: {y_pred_classes.numpy()}")

if __name__ == '__main__':
    # python numpy感知机原始形式
    perceptron_python_demo(10)
    # python numpy感知机对偶形式
    perceptron_duality_python_demo()
    # scikit-learn感知机模型
    perceptron_scikit_learn_demo()
    # tensorflow感知机模型
    perceptron_tensorflow_demo()
    # pytorch感知机模型
    perceptron_pytorch_demo()

