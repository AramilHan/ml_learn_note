# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/2/26 17:40
@File    : sequential_linear_list.py
@Description : Python实现顺序存储结构线性表
"""

class SeqList:
    def __init__(self, size=10):
        """
        初始化线性表，声明三要素：
        1、存储空间的起始位置数组data；
        2、线性表的最大存储容量：MAXSIZE；
        3、线性表的当前长度：length。
        """
        self.max_size = size  # 线性表的最大容量
        self.elements = [None] * size  # 初始化元素数组
        self.length = 0  # 线性表当前长度

    def list_empty(self):
        """
        判断线性表是否为空
        :return: True 或 False
        """
        return self.length == 0

    def list_full(self):
        """
        判断线性表是否已满
        :return: True 或 False
        """
        return self.length == self.max_size

    def add(self, value):
        """
        在线性表末尾添加元素
        :param value: 需要添加的元素
        """
        if self.list_full():
            print("线性表已满，无法添加元素")
            return
        self.elements[self.length] = value
        self.length += 1

    def insert(self, index, value):
        """
        在线性表指定位置插入元素
        :param index: 指定插入的位置
        :param value: 需要插入的元素
        """
        if self.list_full():
            print("线性表已满，无法添加元素")
            return
        if index < 0 or index >= self.length:
            print("插入位置不合法")
            return
        if index <= self.length:
            for i in range(self.length, index, -1):
                self.elements[i] = self.elements[i - 1]
        self.elements[index] = value
        self.length += 1

    def delete(self, index):
        """
        删除线性表指定位置的元素
        :param index: 指定位置
        """
        if self.length == 0:
            print("线性表为空")
            return
        if index < 0 or index >= self.length:
            print("删除位置不合法")
            return
        for i in range(index, self.length - 1):
            self.elements[i] = self.elements[i + 1]
        self.elements[self.length - 1] = None
        self.length -= 1

    def find(self, index):
        """
        :param index: 查询元素
        :return: 元素值
        """
        if self.length == 0 or index < 0 or index >= self.length:
            print("查询元素不合法")
            return None
        value = self.elements[index]
        return value

    def display(self):
        """
        显式线性表中的所有元素
        :return: 表中的元素
        """
        return "->".join(f"{self.elements[i]}" for i in range(self.length))


def main():
    seq_list = SeqList(10)
    seq_list.add(1)
    seq_list.add(2)
    seq_list.insert(1, 4)
    print(f"删除前：{seq_list.display()}")
    seq_list.delete(1)
    print(f"删除后：{seq_list.display()}")


if __name__ == "__main__":
    exit(main())
