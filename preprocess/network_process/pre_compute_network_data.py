#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/2/21
import os
import pickle
import array
import networkx as nx
import numpy as np
import osmnx as ox

from setting import GEO_NAME
from setting import TIME_SLOT
from setting import DISTANCE_EPS, VEHICLE_SPEED
from setting import GEO_DATA_FILE
from utility import print_execute_time
from utility import is_enough_small


@print_execute_time
def compute_shortest_distance():
    """
    shortest_distance 用于查询任意两点之间的最短路径长度 单位长度m
    1. i==j, shortest_distance[i,j] = 0;
    2. i不可以到达j, shortest_distance[i, j] = np.inf
    """
    shortest_distance = np.ones(shape=(node_number, node_number), dtype=np.float32) * np.inf
    for lengths in nx.all_pairs_dijkstra_path_length(graph, weight="length"):
        index_i = osm_id2index[lengths[0]]
        for osm_id_j, distance in lengths[1].items():
            index_j = osm_id2index[osm_id_j]
            shortest_distance[index_i, index_j] = distance
        print(index_i)
    np.save(shortest_distance_file, shortest_distance)


@print_execute_time
def compute_shortest_path():
    """
    shortest_path 记录两点按照最短路径走下一步会到哪个节点
    1. shortest_distance[i, j] == 0.0, shortest_path[i, j] = -1;
    2. shortest_distance[i, j] == np.inf, shortest_path[i, j] = -2;
    """
    shortest_distance = np.load(shortest_distance_file)
    shortest_path = np.ones(shape=(node_number, node_number), dtype=np.int16) * -2
    access_index = [array.array('h') for _ in range(node_number)]  # 可以到达的节点
    adjacent_index = [array.array('h') for _ in range(node_number)]  # 周围相邻的节点
    for paths in nx.all_pairs_dijkstra_path(graph, weight="length"):
        index_i = osm_id2index[paths[0]]
        for osm_id_j, path in paths[1].items():
            index_j = osm_id2index[osm_id_j]
            if index_i == index_j:
                shortest_path[index_i, index_j] = -1
            elif shortest_distance[index_i, index_j] == np.inf:
                shortest_path[index_i, index_j] = -2
            else:
                next_index = osm_id2index[path[1]]
                shortest_path[index_i, index_j] = next_index
                access_index[index_i].append(index_j)
                adjacent_index[index_i].append(osm_id2index[path[1]])

        print(index_i)
    np.save(shortest_path_file, shortest_path)
    with open(access_index_file, "wb") as file1:
        pickle.dump(access_index, file1)
    with open(adjacent_index_file, "wb") as file2:
        pickle.dump(adjacent_index, file2)


@print_execute_time
def compute_shortest_path_time_slot():
    """
    这些文件用于车辆随机更新
    :return:
    """
    shortest_distance = np.load(shortest_distance_file)
    adjacent_location_osm_index = [array.array('h') for _ in range(node_number)]  # 一个时间间隔可以到达的节点
    adjacent_location_driven_distance = [array.array('f') for _ in range(node_number)]  # 一个时间间隔到达某一个节点之后还多行驶的一段距离
    adjacent_location_goal_index = [array.array('h') for _ in range(node_number)]  # 一个时间间隔到达某一个节点之后多行驶朝向方向

    if not os.path.exists(base_file+"/{0}".format(TIME_SLOT)):
        os.mkdir(base_file+"/{0}".format(TIME_SLOT))

    for paths in nx.all_pairs_dijkstra_path(graph, weight="length"):
        index_i = osm_id2index[paths[0]]
        for osm_id_j, path in paths[1].items():
            index_j = osm_id2index[osm_id_j]
            if index_i == index_j or shortest_distance[index_i, index_j] == np.inf:  # 1. 压根没有后续节点的情况
                continue
            if not is_enough_small(could_drive_distance - shortest_distance[index_i, index_j], DISTANCE_EPS):
                # 2. 由于index_i 到 index_j的距离太小了车不会停在index_j上
                continue

            simulated_could_drive_distance = could_drive_distance  # 用于模拟车辆行驶
            for prv_path_osm_id, cur_path_osm_id in zip(path[:-1], path[1:]):
                prv_path_index = osm_id2index[prv_path_osm_id]
                cur_path_index = osm_id2index[cur_path_osm_id]
                two_index_distance = shortest_distance[prv_path_index, cur_path_index]

                if is_enough_small(two_index_distance - simulated_could_drive_distance, DISTANCE_EPS):
                    simulated_could_drive_distance -= two_index_distance
                    if is_enough_small(simulated_could_drive_distance, DISTANCE_EPS):
                        adjacent_location_osm_index[index_i].append(cur_path_index)
                        adjacent_location_driven_distance[index_i].append(0.0)
                        adjacent_location_goal_index[index_i].append(cur_path_index)
                        break
                else:
                    adjacent_location_osm_index[index_i].append(prv_path_index)
                    adjacent_location_driven_distance[index_i].append(simulated_could_drive_distance)
                    adjacent_location_goal_index[index_i].append(cur_path_index)
                    break
        print(index_i)
    with open(adjacent_location_osm_index_file, "wb") as file3:
        pickle.dump(adjacent_location_osm_index, file3)
    with open(adjacent_location_driven_distance_file, "wb") as file4:
        pickle.dump(adjacent_location_driven_distance, file4)
    with open(adjacent_location_goal_index_file, "wb") as file5:
        pickle.dump(adjacent_location_goal_index, file5)


if __name__ == '__main__':
    could_drive_distance = VEHICLE_SPEED * TIME_SLOT  # 车辆在一个时间间隔内可以行驶的距离
    base_file = "../../data/{0}/network_data/".format(GEO_NAME)
    graph_file = GEO_DATA_FILE["graph_file"]
    osm_id2index_file = os.path.join(base_file, GEO_DATA_FILE["osm_id2index_file"])
    index2osm_id_file = os.path.join(base_file, GEO_DATA_FILE["index2osm_id_file"])
    shortest_distance_file = os.path.join(base_file, GEO_DATA_FILE["shortest_distance_file"])
    shortest_path_file = os.path.join(base_file, GEO_DATA_FILE["shortest_path_file"])
    access_index_file = os.path.join(base_file, GEO_DATA_FILE["access_index_file"])
    adjacent_index_file = os.path.join(base_file, GEO_DATA_FILE["adjacent_index_file"])
    adjacent_location_osm_index_file = os.path.join(base_file, GEO_DATA_FILE["adjacent_location_osm_index_file"])
    adjacent_location_driven_distance_file = os.path.join(base_file, GEO_DATA_FILE["adjacent_location_driven_distance_file"])
    adjacent_location_goal_index_file = os.path.join(base_file, GEO_DATA_FILE["adjacent_location_goal_index_file"])

    # load raw data
    graph = ox.load_graphml(filename=graph_file, folder="../raw_data/{0}_raw_data/".format(GEO_NAME))

    # osm_id2index = {}
    # index2osm_id = {}
    # for i, osm_id in enumerate(graph.nodes):
    #     osm_id2index[osm_id] = i
    #     index2osm_id[i] = osm_id
    #
    # with open(index2osm_id_file, "wb") as file:
    #     pickle.dump(index2osm_id, file)
    #
    # with open(osm_id2index_file, "wb") as file:
    #     pickle.dump(osm_id2index, file)

    with open(osm_id2index_file, "rb") as file:
        osm_id2index = pickle.load(file)
    with open(index2osm_id_file, "rb") as file:
        index2osm_id = pickle.load(file)
    node_number = len(osm_id2index)

    # 计算最短路径长度矩阵
    compute_shortest_distance()

    # 得到下到一个节点的需要经过的节点
    # compute_shortest_path()

    # 计算到一个节点过程中一分钟可以到哪些节点
    # compute_shortest_path_time_slot()
