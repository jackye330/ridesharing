#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/2/22
import time

from algorithm.matching import *
from setting import *
from utility import *


def order_matching_simulation(shortest_distance, shortest_path, shortest_path_with_minute, orders, vehicles, method):
    social_welfare_result = []
    social_cost_result = []
    total_profit_result = []
    total_payment_result = []
    total_utility_result = []
    unchange_vehicle_result = []
    empty_vehicle_rate = []
    un_matched_orders = set()
    for t in range(MIN_REQUEST_TIME, MAX_REQUEST_TIME):
        # 将上一轮没有分配的订单和当前已经发布的订单合并
        un_matched_orders = update_orders(un_matched_orders, orders[t], t)
        if method.__name__ in ['orders_matching_with_vcg', 'orders_matching_with_gm']:
            bids = generate_bids(shortest_distance, un_matched_orders, vehicles, t)  # 投标
            match_result = method(bids)  # 匹配
        else:
            match_result = method(shortest_distance, un_matched_orders, vehicles, t)  # 匹配
        # 记录结果
        matched_orders, payments, social_welfare, social_cost, total_payment, total_utility, total_profit = match_result
        social_welfare_result.append(social_welfare)
        social_cost_result.append(social_cost)
        total_payment_result.append(total_payment)
        total_utility_result.append(total_utility)
        total_profit_result.append(total_profit)
        form_location = {}
        for vehicle in vehicles:
            form_location[vehicle] = vehicle.location.osm_index
        print(social_welfare, total_profit, total_utility)
        # 车辆更新位置
        update_vehicle_location(shortest_distance, shortest_path, shortest_path_with_minute, vehicles, payments, t)
        # 平台进行分配并将没有分配的订单存入一个
        unchange_vehicle_num = 0
        for vehicle in vehicles:
            if form_location[vehicle] == vehicle.location.osm_index:
                unchange_vehicle_num += 1
        unchange_vehicle_result.append(unchange_vehicle_num/len(vehicles))
        empty_vehicle_num = 0
        for vehicle in vehicles:
            # 统计空闲车的数量
            if vehicle.status == vehicle.WITHOUT_MISSION_STATUS:
                empty_vehicle_num += 1
        empty_vehicle_rate.append(empty_vehicle_num / len(vehicles))
        un_matched_orders = un_matched_orders - matched_orders
        print(t)
    return social_welfare_result, social_cost_result, total_payment_result, total_utility_result, total_profit_result, unchange_vehicle_result, empty_vehicle_rate


def order_dispatch_simulation(shortest_distance, shortest_path, shortest_path_with_minute,
                              orders, vehicles, method):
    social_welfare_result = []
    social_cost_result = []
    total_profit_result = []
    total_payment_result = []
    total_utility_result = []
    un_dispatched_orders = set()
    for t in range(MIN_REQUEST_TIME, MAX_REQUEST_TIME):
        # 将上一轮没有分配的订单和当前已经发布的订单合并
        un_dispatched_orders = update_orders(un_dispatched_orders, orders[t], t)
        # vehicle根据自己的情况进行投标并平台计算最优的分配并计算支付价格
        dispatch_result = method(shortest_distance, shortest_path, shortest_path_with_minute, orders, vehicles, t)
        # 记录结果
        dispatched_orders, payments, social_welfare, social_cost, total_payment, total_utility, total_profit = dispatch_result
        social_welfare_result.append(social_welfare)
        social_cost_result.append(social_cost)
        total_payment_result.append(total_payment)
        total_utility_result.append(total_utility)
        total_profit_result.append(total_profit)
        # 平台进行分配并将没有分配的订单存入一个
        un_dispatched_orders = un_dispatched_orders - dispatched_orders
        print(t)
    return social_welfare_result, social_cost_result, total_payment_result, total_utility_result, total_profit_result


if __name__ == '__main__':
    with open("./env_data/static.pkl", "rb") as file:
        static = pickle.load(file)
        graph, shortest_distance, shortest_path, shortest_path_with_minute = static

    for k in range(MIN_REPEATS, MAX_REPEATS):
        with open("./env_data/vehicle/{0}_{1}.pkl".format(VEHICLE_NUMBER, k), "rb") as file:
            vehicles = pickle.load(file)
        with open("./env_data/order/{0}_{1}.pkl".format(VEHICLE_NUMBER, k), "rb") as file:
            orders = pickle.load(file)
        env_data = shortest_distance, shortest_path, shortest_path_with_minute, orders, vehicles, orders_matching_with_vcg
        env_result = order_matching_simulation(*env_data)
        served_order_number, total_order_number = 0, 0
        for t in range(MIN_REQUEST_TIME, MAX_REQUEST_TIME):
            for order in orders[t]:
                if order.belong_to_vehicle:
                    served_order_number += 1
                total_order_number += 1
        service_rate = served_order_number / total_order_number
        result = (*env_result, service_rate)
        with open("./{0}_{1}.pkl".format(VEHICLE_NUMBER, k), "wb") as file:
            pickle.dump(obj=result, file=file)
