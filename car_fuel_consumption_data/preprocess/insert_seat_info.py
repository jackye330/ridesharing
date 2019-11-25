#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/2/21
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

car_info = pd.read_csv("./car_fuel_consumption_info.csv", encoding="gb18030")
car_info = car_info[car_info["fuel_consumption"] >= 4.1]
n = car_info.shape[0]
car_info = car_info.sort_values(by="weight")
seats = np.zeros(shape=(n,), dtype=np.int8)
car_info["seats"] = seats
car_info["weight"].hist()
plt.show()
# car_info[car_info["fuel"]]
# car_info.to_csv("./car_fuel_consumption_info.csv", index=False)
