#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/7/7
from setting import *
from utility import *

vehicle_data_dir = "./env_data/vehicle"
order_data_dir = "./env_data/order"
env_data_dir = "./env_data/static.pkl"
# 保存数据
if __name__ == '__main__':

    for i in range(0, 10):
        env_parameters = MIN_REQUEST_TIME, MAX_REQUEST_TIME, MAX_WAIT_TIMES, DETOUR_RATIOS,\
                         VEHICLE_NUMBER, AVERAGE_SPEED
        env_data = initialize_environment(*env_parameters)
        graph, shortest_distance, shortest_path, shortest_path_with_minute, adjacent_nodes, orders, vehicles = env_data

        with open("{0}/{1}_{2}.pkl".format(vehicle_data_dir, VEHICLE_NUMBER, i), "wb") as file:
            pickle.dump(obj=vehicles, file=file)
        with open("{0}/{1}_{2}.pkl".format(order_data_dir, VEHICLE_NUMBER, i), "wb") as file:
            pickle.dump(obj=orders, file=file)
    with open(env_data_dir, "wb") as file:
        static = graph, shortest_distance, shortest_path, shortest_path_with_minute, adjacent_nodes
        pickle.dump(static, file)

