#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/28
import osmnx as ox
import pandas as pd
from setting import FLOAT_ZERO
from setting import ORDER_DATA_FILES
from setting import HaiKou, Manhattan
"""
直接通过已经筛选好的订单数据里面的运行历程除去时间计算平均值得到
"""
average_vehicle_speed = FLOAT_ZERO
for file_name in ORDER_DATA_FILES:
    csv_data = pd.read_csv(file_name)
    vehicle_speed = csv_data["order_distance"] / csv_data["order_time"]
    average_vehicle_speed += vehicle_speed.mean()

print(average_vehicle_speed / len(ORDER_DATA_FILES))

"""
理论的最大值
"""
G = ox.load_graphml("{0}.graphml".format(Manhattan), "../data/{0}/network_data/".format(Manhattan))
speed_set = set()
for edge in G.edges(data=True):
    if "maxspeed" in edge[2]:
        print(edge[2]["maxspeed"])
        speed_set.add(edge[2]["maxspeed"])
print(speed_set)
