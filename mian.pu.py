#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/9/5
import pickle
s = 0
with open("./env_data/order/1500_0.pkl", "rb") as f:
    data = pickle.load(f)
    for x in data.items():
        s += len(x)
print(s)