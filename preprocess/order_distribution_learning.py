#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/28
"""
这里我们认为订单到来满足泊松过程，具体学习过程可以参考论文  T-Share: A Large-Scale Dynamic Taxi Ridesharing Service (ICDE2013)
位置p，在t时刻，需要学习会有多少订单生成N_p^t, 然后转移矩阵A_p_q^t， 就是t时段从p到q的转移概率，还有t时段单位费率f_t在这一时段单位价钱
我们将一天分成了24个时间段，每一个时间段的过程统计起始点分布，终止点分布，还有平均价格
"""
import os
import pickle
import numpy as np
import pandas as pd
from setting import GEO_NAME
from setting import ORDER_DATA_FILES
from setting import GEO_DATA_FILE
input_geo_data_file_dir = "../data/{0}/network_data".format(GEO_NAME)
with open(os.path.join(input_geo_data_file_dir, GEO_DATA_FILE["index2osm_id_file"]), "rb") as file:
    index2osm_id = pickle.load(file)

lambda_i_t = np.zeros(shape=(24, len(index2osm_id)), dtype=np.int32)  # 起始点i的泊松过程的参数
lambda_j_t = np.zeros(shape=(24, len(index2osm_id)), dtype=np.int32)  # 结束点j的泊松过程的参数
unit_fare_t = np.zeros(shape=(24,))  # 单位价格分布

for hour in range(24):
    for i in range(30):
        data_file = pd.read_csv("../data/{}")