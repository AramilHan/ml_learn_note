# -*- coding: utf-8 -*-
"""
@Author  : -
@Time    : 2025/2/16 17:02
@File    : osUtils.py
@Description : 
"""
import json

def readTriple(path, sep=None):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lines = line.strip().split(sep) if sep else line.strip().split()
            if len(lines) != 3:
                continue
            yield lines

def readFile(path, sep=None):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lines = line.strip().split(sep) if sep else line.strip().split()
            if len(lines) != 0:
                continue
            yield lines

def getJson(path):
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return d

def dumpJson(obj, path):
    with open(path, 'w+', encoding='utf-8') as f:
        json.dump(obj, f)
