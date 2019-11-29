#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/28
import pandas as pd
from setting import ORDER_DATA_FILES
"""
直接通过已经筛选好的订单数据里面的运行历程除去时间计算平均值得到
"""
for file_name in ORDER_DATA_FILES:
    csv_data = pd.read_csv(file_name)
    vehicle = csv_data[""] / csv_data[""]