#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/2/21
import pickle
import networkx as nx
import numpy as np
import osmnx as ox

if __name__ == '__main__':
    average_speed = 1.609344 * 12 / 60
    graph = ox.load_graphml(filename="Manhattan.graphml", folder="./")
    nodes = list(graph.nodes)
    node_number = len(nodes)
    osm_id2index = {}
    index2osm_id = {}
    for i, osm_id in enumerate(graph.nodes):
        osm_id2index[osm_id] = i
        index2osm_id[i] = osm_id

    with open("./osm_id2index.pkl", "wb") as file:
        pickle.dump(osm_id2index, file)

    # 用于查询任意两点之间的最短路径，记录任意两点之间的距离并且记录
    # 1. i==j, shortest_length[i,j] = 0;
    # 2. i不可以到达j, shortest_length[i, j] = np.inf

    shortest_distance = np.ones(shape=(node_number, node_number), dtype=np.float32) * np.inf
    for lengths in nx.all_pairs_bellman_ford_path_length(graph, weight="length"):
        i = osm_id2index[lengths[0]]
        for node_j, length in lengths[1].items():
            j = osm_id2index[node_j]
            shortest_distance[i, j] = length / 1000.0
        print(i)

    np.save("./shortest_distance.npy", shortest_distance)

    # 用于车辆随机行走的查询下一分钟会到达的节点
    # 1. shortest_length[i, j]太小, shortest_path_with_minute[i, j] = -1;
    # 2. shortest_length[i, j]太大, shortest_path_with_minute[i, j] = -2;
    # 3. 如果存在车辆1min钟路程刚好可以跨越两个半相邻的节点那么取最接近1min路程的节点
    shortest_path_with_minute = np.ones(shape=(node_number, node_number), dtype=np.int16) * -2
    for paths in nx.all_pairs_bellman_ford_path(graph, weight="length"):
        i = osm_id2index[paths[0]]
        for node_j, path in paths[1].items():
            j = osm_id2index[node_j]

            current_trip_distance = 0.0   # 当前模拟车辆的行程
            if average_speed - shortest_distance[i, j] > 0.01:  # 首先判断是否可以一分钟是否过剩
                shortest_path_with_minute[i, j] = -1
            elif shortest_distance[i, j] == np.inf:  # 然后判断是否可达
                shortest_path_with_minute[i, j] = -2
            else:
                for previous_location, current_location in zip(path[:-1], path[1:]):
                    previous_location_index = osm_id2index[previous_location]
                    current_location_index = osm_id2index[current_location]
                    two_location_distance = shortest_distance[previous_location_index, current_location_index]

                    if current_trip_distance + two_location_distance - average_speed < -0.01:
                        current_trip_distance += two_location_distance
                    elif -0.01 <= current_trip_distance + two_location_distance - average_speed <= 0.01:
                        shortest_path_with_minute[i, j] = current_location_index
                        break
                    else:
                        # current_trip_distance + two_location_distance - average_speed <
                        # average_speed - current_trip_distance:
                        # 为了代码篇幅写成下面的形式
                        if 2 * current_trip_distance + two_location_distance < 2 * average_speed:  # 取最接近1min路程的节点
                            shortest_path_with_minute[i, j] = current_location_index
                        else:
                            shortest_path_with_minute[i, j] = previous_location_index
                        break

        print(i)
    np.save("./shortest_path_with_minute.npy", shortest_path_with_minute)

    shortest_distance = np.load("./shortest_distance.npy")
    # 记录两点按照最短路径走下一步会到哪个节点
    # 1. shortest_length[i, j] == 0.0, shortest_path[i, j] = -1;
    # 2. shortest_length[i, j] == np.inf, shortest_path[i, j] = -2;

    shortest_path = np.ones(shape=(node_number, node_number), dtype=np.int16) * -2
    for paths in nx.all_pairs_bellman_ford_path(graph, weight="length"):
        i = osm_id2index[paths[0]]
        for node_j, path in paths[1].items():
            j = osm_id2index[node_j]
            if i == j:
                shortest_path[i, j] = -1
            elif shortest_distance[i, j] == np.inf:
                shortest_path[i, j] = -2
            else:
                shortest_path[i, j] = osm_id2index[path[1]]
        print(i)
    np.save("./shortest_path.npy", shortest_path)




