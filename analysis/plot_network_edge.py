#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/4
import os
import numpy as np
import matplotlib.pyplot as plt
from setting import GEO_DATA_FILE
base_folder = "../data/network_data"
shortest_distance_file = os.path.join(base_folder, GEO_DATA_FILE["shortest_distance_file"])
shortest_path_file = os.path.join(base_folder, GEO_DATA_FILE["shortest_path_file"])
shortest_distance = np.load(shortest_distance_file)
shortest_path = np.load(shortest_path_file)
edges = []
n = len(shortest_distance)
for i in range(n):
    for j in range(n):
        if shortest_distance[i, j] != np.inf and i != j and shortest_path[i, j] == j:
            edges.append(shortest_distance[i, j])
plt.hist(edges, bins=30, normed=True)
plt.show()

# 绘制图像看最长的道路
# shortest_distance = np.load(shortest_distance_file)
# shortest_path = np.load(shortest_path_file)
# max_l = -np.inf
# max_i = 0
# max_j = 0
# for i in range(node_number):
#     for j in range(node_number):
#         if shortest_path[i, j] == j and i != j and shortest_distance[i, j] != np.inf and max_l < shortest_distance[i, j]:
#             max_i = i
#             max_j = j
#             max_l = shortest_distance[i, j]
# print(max_l, max_i, max_j)
# origin = index2osm_id[max_i]
# destination = index2osm_id[max_j]
# bid_route = nx.shortest_path(graph, origin, destination)
# ox.plot_graph_route(graph, bid_route)
