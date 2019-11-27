#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/6/26
import os
import pickle
import numpy as np
import osmnx as ox
import pandas as pd
from setting import GEO_NAME
from setting import GEO_DATA_FILE
result_dir = "../data/{0}/order_data".format(GEO_NAME)
temp_dir = "./raw_data/temp/{0}/".format(GEO_NAME)
green = "green"
yellow = "yellow"
color = yellow
chunk_size = 10000
eps = 0.0003  # 点对齐的精度
raw_order_filename = os.path.join("./raw_data/NewYork_raw_data", "{0}_trip_data_2016-06.csv".format(color))
temp_result_file1 = os.path.join(temp_dir, "{0}_temp1.csv".format(color))
temp_result_file2 = os.path.join(temp_dir, "{0}_temp2.csv".format(color))


def time2int(time):
    _s = time.split(":")
    return str(int(_s[0]) * 60 * 60 + int(_s[1]) * 60 + int(_s[2]))


def list2str(lis):
    return ",".join(list(map(str, lis)))


if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)

# 剔除异常数据
cnt = 0
temp_file = open(temp_result_file1, "w")
temp_file.write(",".join(["day", "pick_time", "pick_longitude", "pick_latitude", "drop_longitude", "drop_latitude",
                          "passenger_count", "trip_distance", "fare_amount", "tip_amount", "total_amount"]) + "\n")
for csv_iterator in pd.read_table(raw_order_filename, chunksize=chunk_size, iterator=True):
    for line in csv_iterator.values:
        s = line[0].split(',')
        if color == green:
            if s[5] == '0' or s[6] == '0' or s[7] == '0' or s[8] == '0' or s[9] == '0' or \
                    s[10] == '0' or float(s[11]) <= 0 or float(s[10]) < 0.5 or float(s[11]) > 10 * float(s[10]):
                pass
            else:
                date = s[1].split(" ")[0].split("-")[-1]
                s[1] = time2int(s[1].split(" ")[1])
                s = s[1:2] + s[5:12] + s[14:15] + s[18:19]
                temp_file.write(date + "," + ",".join(s) + "\n")
                cnt += 1
        else:
            if s[5] == '0' or s[6] == '0' or s[9] == '0' or s[10] == '0' or s[3] == '0' or \
                    s[4] == '0' or float(s[12]) <= 0 or float(s[4]) < 0.5 or float(s[12]) > 10 * float(s[4]):
                pass
            else:
                date = s[1].split(" ")[0].split("-")[-1]
                s[1] = time2int(s[1].split(" ")[1])
                s = s[1:2] + s[5:7] + s[9:11] + s[3:5] + s[12:13] + s[15:16] + s[18:19]
                temp_file.write(date + "," + ",".join(s) + "\n")
                cnt += 1

print(cnt)
temp_file.close()

# 点对齐
G = ox.load_graphml(GEO_NAME + ".graph" + "ml", "../data/{0}/network_data".format(GEO_NAME))  # 注意：".graph"+"ml" 是不为了飘绿色
nodes = G.nodes(data=True)
nodes_dict = {node[0]: (float(node[1]['x']), float(node[1]['y'])) for node in nodes}

temp_file = open(temp_result_file2, "w")
temp_file.write(",".join(["day", "pick_time", "pick_som_id", "drop_osm_id", "passenger_count",
                          "trip_distance", "fare_amount", "tip_amount", "total_amount"]) + "\n")

cnt = 0
for csv_iterator in pd.read_table(temp_result_file1, chunksize=chunk_size, iterator=True):
    data = []
    for line in csv_iterator.values:
        data.append(list(map(float, line[0].split(","))))
    data = np.array(data)
    pick_lon = data[:, 2]
    pick_lat = data[:, 3]
    drop_lon = data[:, 4]
    drop_lat = data[:, 5]
    correct_pick_osm_ids = ox.get_nearest_nodes(G, pick_lon, pick_lat, method="balltree")
    correct_drop_osm_ids = ox.get_nearest_nodes(G, drop_lon, drop_lat, method="balltree")
    for idx in range(len(correct_pick_osm_ids)):
        correct_pick_lon = nodes_dict[correct_pick_osm_ids[idx]][0]
        correct_pick_lat = nodes_dict[correct_pick_osm_ids[idx]][1]
        correct_drop_lon = nodes_dict[correct_drop_osm_ids[idx]][0]
        correct_drop_lat = nodes_dict[correct_drop_osm_ids[idx]][1]
        correct_corr = np.array([correct_pick_lon, correct_pick_lat, correct_drop_lon, correct_drop_lat])
        if (np.abs(data[idx, 2:6] - correct_corr) <= eps).all():
            temp_list = data[idx, :2].tolist() + \
                        [correct_pick_osm_ids[idx], correct_drop_osm_ids[idx]] + \
                        data[idx, 6:].tolist()
            temp_file.write(list2str(temp_list) + "\n")
            cnt += 1
        if cnt % chunk_size == 0:
            print(cnt)
print(cnt)
temp_file.close()

# 合并黄绿出租车的订单数据
temp_green_result_file = os.path.join(temp_dir, "green_temp2.csv")
temp_yellow_result_file = os.path.join(temp_dir, "yellow_temp2.csv")
osm_id2index_file = GEO_DATA_FILE["osm_id2index_file"]
with open(os.path.join("../data/{0}/network_data/".format(GEO_NAME), osm_id2index_file), "rb") as file:
    osm_id2index = pickle.load(file)
order_files = []
for i in range(30):
    file = open(os.path.join(temp_dir, "order_data_2016_06_{0:2}.csv".format(i + 1)), "w")
    file.write(
        "pick_time,pick_index,drop_index,n_riders,order_distance,order_fare,order_tip,total_fare\n")
    order_files.append(file)

for csv_iterator in pd.read_table(temp_green_result_file, chunksize=chunk_size, iterator=True):
    for line in csv_iterator.values:
        s = line[0].split(",")
        index = int(float(s[0])) - 1
        data = s[1:]
        data[0] = str(int(float(data[0])))
        data[1] = str(osm_id2index[int(data[1])])
        data[2] = str(osm_id2index[int(data[2])])
        data[3] = str(int(float(data[3])))
        order_files[index].write(",".join(data) + '\n')

for csv_iterator in pd.read_table(temp_yellow_result_file, chunksize=chunk_size, iterator=True):
    for line in csv_iterator.values:
        s = line[0].split(",")
        index = int(float(s[0])) - 1
        data = s[1:]
        data[0] = str(int(float(data[0])))
        data[1] = str(osm_id2index[int(data[1])])
        data[2] = str(osm_id2index[int(data[2])])
        data[3] = str(int(float(data[3])))
        order_files[index].write(",".join(data) + '\n')
for i in range(30):
    order_files[i].close()

# 按时间排序
for i in range(30):
    df_ = pd.read_csv(os.path.join(temp_dir, "order_data_2016_06_{0:2}.csv".format(i + 1)))
    df_.sort_values(by=["pick_time"], inplace=True)
    df_.to_csv(os.path.join(result_dir, "order_data_2016-06-{0:02}.csv".format(i + 1)), sep=',', index=False)

for i in range(30):
    os.rename(os.path.join(result_dir, "order_data_2016-06-{0:02}.csv".format(i + 1)),
              os.path.join(result_dir, "order_data_{0:03}.csv".format(i)))
