#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/7/8
import pickle
import time
from simulation import *
import matplotlib.pyplot as plt

from algorithm.simple_dispatching import *
from setting import *


def run(method):
    running_time_result = []
    for k in range(1):
        with open("../env_data/vehicle/{0}_{1}.pkl".format(VEHICLE_NUMBER, k), "rb") as file:
            vehicles = pickle.load(file)
        with open("../env_data/order/{0}_{1}.pkl".format(VEHICLE_NUMBER, k), "rb") as file:
            orders = pickle.load(file)
        with open("../env_data/static.pkl", "rb") as file:
            static = pickle.load(file)
            graph, shortest_distance, shortest_path, shortest_path_with_minute = static
        epoch_time = []
        un_matched_orders = set()
        for t in range(MIN_REQUEST_TIME, MAX_REQUEST_TIME):
            # 将上一轮没有分配的订单和当前已经发布的订单合并
            un_matched_orders = update_orders(un_matched_orders, orders[t], t)
            if method.__name__ in ['orders_matching_with_vcg', 'orders_matching_with_gm']:
                bids = generate_bids(shortest_distance, un_matched_orders, vehicles, t)  # 投标
                t1 = time.time()
                match_result = method(bids)  # 匹配
            else:
                t1 = time.time()
                match_result = method(shortest_distance, un_matched_orders, vehicles, t)  # 匹配
            t2 = time.time()
            epoch_time.append(t2 - t1)
            # 记录结果
            matched_orders, payments, social_welfare, social_cost, total_payment, total_utility, total_profit = match_result
            # 车辆更新位置
            update_vehicle_location(shortest_distance, shortest_path, shortest_path_with_minute, vehicles, payments, t)
            # 平台进行分配并将没有分配的订单存入一个
            un_matched_orders = un_matched_orders - matched_orders
            print(t)
        running_time_result.append(epoch_time)
    with open("../result/running_time/{0}_{1}.pkl".format(method.__name__, VEHICLE_NUMBER), "wb") as file:
        pickle.dump(running_time_result, file)


if __name__ == '__main__':
    ### 时间计算 ####
    # for method in [orders_matching_with_nearest_matching, orders_matching_with_gm, orders_matching_with_vcg]:
    #     run(method)
    ### 绘图分析 ####
    plt.rc('font', family='Times New Roman', weight=3)
    model_name = ["Nearest-Matching", "MSWR-VCG", "MSWR-GM"]
    style = ['g-v', 'r-s', 'b-^']
    # model_name = ["Nearest-Matching", "SWMOM-VCG", "SWMOM-GM", "SPARP"]
    # style = ['g-v', 'r-s', 'b-^', 'y-o']
    min_x, max_x, step_x = 500, 2501, 500
    label = range(min_x, max_x, step_x)
    legend_fontsize = 14
    label_fontsize = 16
    plt.figure(figsize=(7, 6))
    for index, method in enumerate(
            [orders_matching_with_nearest_matching, orders_matching_with_vcg, orders_matching_with_gm]):
        data_sum = []
        for vehicle_number in range(500, 2501, 500):
            with open("../result/running_time/{0}_{1}.pkl".format(method.__name__, vehicle_number), "rb") as file:
                data = pickle.load(file)
            data = np.array(data)
            data_sum.append(np.sum(data, axis=1))
        data_sum = np.array(data_sum)
        print(data_sum)
        data_sum = data_sum.mean(axis=1)
        ind = range(len(data_sum))
        plt.plot(ind, data_sum, style[index], linewidth=3, markersize=10, markerfacecolor='w', markeredgewidth=3)
    plt.xticks(ind, label)
    plt.legend(model_name, fontsize=legend_fontsize)
    plt.grid(True)
    plt.xlabel("#Vehicles", fontsize=label_fontsize)
    plt.ylabel("Running Time (s)", fontsize=label_fontsize)
    plt.show()

