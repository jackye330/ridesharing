#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/2/22
import time
from algorithm.order_matching import *
from setting import *
from utility import *


def order_matching_simulation(shortest_distance, shortest_path, shortest_path_with_minute, orders, vehicles, method):
    social_welfare_result = []
    social_cost_result = []
    total_profit_result = []
    total_payment_result = []
    total_utility_result = []
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
        print(social_welfare, total_profit, total_utility)
        # 车辆更新位置
        update_vehicle_location(shortest_distance, shortest_path, shortest_path_with_minute, vehicles, payments, t)
        # 平台进行分配并将没有分配的订单存入一个
        un_matched_orders = un_matched_orders - matched_orders
        print(t)
    return social_welfare_result, social_cost_result, total_payment_result, total_utility_result, total_profit_result


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


def order_dispatch_one_simulation(shortest_distance, shortest_path, shortest_path_with_minute,
                              orders, vehicles):
    social_welfare_result = []
    social_cost_result = []
    total_profit_result = []
    total_payment_result = []
    total_utility_result = []
    served_order = 0
    # 还是按照一分钟的时间进行车辆位置的更新（一分钟以内车行进距离太短，很容易被修正到原来的位置）
    for t in range(MIN_REQUEST_TIME, MAX_REQUEST_TIME):
        print("round:", t)
        order_t = orders[t]
        social_welfare = 0.0
        social_cost = 0.0
        total_payment = 0.0
        total_utility = 0.0
        total_profit = 0.0
        for order in order_t:
            bids = {}
            costs = {}
            route_plans = {}
            for vehicle in vehicles:    # 计算每个车的投标
                # 超过容积限制或太远
                if vehicle.n_seats < order.n_riders or \
                        shortest_distance[vehicle.location.osm_index][order.start_location.osm_index] > \
                        vehicle.AVERAGE_SPEED * order.max_wait_time:
                    bids[vehicle] = - np.inf
                    costs[vehicle] = - np.inf
                    route_plans[vehicle] = []
                    continue
                rem_list = vehicle.get_remain_list(vehicle.route_plan.copy())  # route_plan要转成只有起始点或单独一个终点的rem_list
                old_profit, old_cost, _ = vehicle.find_best_schedule([], rem_list.copy(), shortest_distance, t, 0.0, 0.0, [])
                rem_list.append(order.start_location)
                # 把这里算的路径存起来
                new_profit, new_cost, new_route_plan = vehicle.find_best_schedule([], rem_list.copy(), shortest_distance, t, 0.0, 0.0, [])
                route_plans[vehicle] = new_route_plan
                costs[vehicle] = new_cost - old_cost    # 这一单的成本
                bids[vehicle] = new_profit - old_profit  # vehicle 投标为 additional_profit
            bids_list = sorted(bids.items(), key=lambda k: k[1], reverse=True)
            profit_ = bids_list[0][1]  # 最大利润 bids_list里的vehicle不是原来哪个vehicle不能直接更新状态
            second_price = bids_list[1][1]   # second_price
            costs_list = sorted(costs.items(), key=lambda k: k[1], reverse=True)
            cost_ = costs_list[0][1]    # 最大的成本
            reserve_price = order.trip_fare - cost_     # reserve_price = 订单原始价格减去成本的最大值，\
            # 如果不高于reserve_price则不分配，且保证second_price 不低于0
            if profit_ < 0 or profit_ < reserve_price:
                continue
            for vehicle in vehicles:
                if bids[vehicle] == profit_:
                    # 更新vehicle_的订单状况
                    # rem_list = vehicle.get_rem_list(vehicle.route_plan.copy())
                    # rem_list.append(order.start_location)
                    # _, route_plan_ = vehicle.find_best_schedule([], rem_list.copy(), shortest_distance,
                    #                                          t, 0.0, [])
                    # print(vehicle.vehicle_id)
                    vehicle.route_plan = route_plans[vehicle]   # 可能会有两单的起始点和终点相同
                    vehicle.status = Vehicle.HAVE_MISSION_STATUS  # 更新车的状态
                    vehicle.n_seats -= order.n_riders
                    order.belong_to_vehicle = vehicle
                    social_welfare += profit_  # 增加payment social_welfare没有变化 sw = payment + fare - cost - payment (payment 为司机向平台支付的金额)
                    social_cost += costs[vehicle]
                    if second_price > 0:
                        total_payment += second_price
                        total_profit += second_price
                        total_utility += profit_ - second_price
                    else:
                        total_payment += reserve_price
                        total_profit += reserve_price
                        total_utility += profit_ - reserve_price
                    break
        #   一个t时间结束再进行车辆位置的更新
        for vehicle in vehicles:
            if vehicle.status == Vehicle.WITHOUT_MISSION_STATUS:
                vehicle.update_random_location(shortest_distance, shortest_path_with_minute)
            else:
                vehicle.update_order_location(shortest_distance, shortest_path)
        social_welfare_result.append(social_welfare)
        social_cost_result.append(social_cost)
        total_payment_result.append(total_payment)
        total_utility_result.append(total_utility)
        total_profit_result.append(total_profit)
    # for order in orders:
    #     if order.request_time - current_time >= 60:
    #         for vehicle in vehicles:
    #             if vehicle.status == Vehicle.WITHOUT_MISSION_STATUS:
    #                 vehicle.update_random_location(shortest_distance, shortest_path, shortest_path_with_minute)
    #             else:
    #                 vehicle.update_order_location(shortest_distance, shortest_path, shortest_path_with_minute)
    #         current_time = order.request_time
    #     bids = {}
    #     for vehicle in vehicles:
    #         # route_plan转换成rem_list
    #         rem_list = vehicle.get_rem_list(vehicle.route_plan)
    #         _, old_profit = vehicle.find_best_schedule([], rem_list, shortest_distance, current_time, 0.0, [])
    #         rem_list.append(order)
    #         _, new_profit = vehicle.find_best_schedule([], rem_list, shortest_distance, current_time, 0.0, [])
    #         bids[vehicle] = new_profit - old_profit  # vehicle 投标为 additional_profit
    #     bids_list = sorted(bids.items(), key=lambda k: k[1], reverse=True)
    #     profit_ = bids_list[0][1]    # bids_list里的vehicle不是原来哪个vehicle不能直接更新状态
    #     for vehicle in vehicles:
    #         if bids[vehicle] == profit_:
    #             # 更新vehicle_的订单状况
    #             rem_list = vehicle.get_rem_list(vehicle.route_plan)
    #             route_plan_ = vehicle.find_best_schedule([], rem_list, shortest_distance,
    #                                                      current_time, 0.0, [])
    #             vehicle.route_plan = route_plan_
    #             vehicle.status = Vehicle.HAVE_MISSION_STATUS    # 更新车的状态
    #             vehicle.seats -= order.n_riders
    #             order.belong_to_vehicle = vehicle
    #             break
    #     social_welfare_result.append(profit_)
    #     social_cost_result.append(order.trip_fare - profit_)    # profit = trip_fare - cost
    #     total_payment_result.append(order.trip_fare - profit_)  # trip_fare要做调整
    #     total_utility_result.append(0)  # 平台向司机支付的就是司机的成本
    #     total_profit_result.append(profit_)
    # print("served_order", served_order)
    return social_welfare_result, social_cost_result, total_payment_result, total_utility_result, total_profit_result





if __name__ == '__main__':
    start_time = time.clock()
    with open("./env_data/static.pkl", "rb") as file:
        static = pickle.load(file)
        graph, shortest_distance, shortest_path, shortest_path_with_minute = static

    for k in range(MIN_REPEATS, MAX_REPEATS):
        print("repeat:", k)
        with open("./env_data/vehicle/{0}_{1}.pkl".format(VEHICLE_NUMBER, k), "rb") as file:
            vehicles = pickle.load(file)
        with open("./env_data/order/{0}_{1}.pkl".format(VEHICLE_NUMBER, k), "rb") as file:
            orders = pickle.load(file)
        env_data = shortest_distance, shortest_path, shortest_path_with_minute, orders, vehicles, orders_matching_with_nearest_matching
        env_result = order_matching_simulation(*env_data)
        # env_data = shortest_distance, shortest_path, shortest_path_with_minute, orders.copy(), vehicles
        # env_result = order_dispatch_one_simulation(*env_data)
        served_order_number, total_order_number = 0, 0
        for t in range(MIN_REQUEST_TIME, MAX_REQUEST_TIME):
            for order in orders[t]:
                if order.belong_to_vehicle:
                    served_order_number += 1
                total_order_number += 1
        # print("total_order", total_order_number)
        service_rate = served_order_number / total_order_number
        # print('ss', served_order_number)
        result = (*env_result, service_rate)
        with open("./result//{0}_{1}.pkl".format(VEHICLE_NUMBER, k), "wb") as file:
            pickle.dump(obj=result, file=file)
    elapsed = (time.clock() - start_time)
    print("time used:", elapsed)
