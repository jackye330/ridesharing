#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : lyk
# date   : 2019/9/10
# 额外计算每分钟的服务率
import time
from algorithm.matching import *
from algorithm.dispatching import *
from setting import *
from utility import *
import copy


def order_matching_simulation(shortest_distance, shortest_path, shortest_path_with_minute, adjacent_nodes, orders, vehicles, method):
    social_welfare_result = []
    social_cost_result = []
    total_profit_result = []
    total_payment_result = []
    total_utility_result = []
    service_rate_result = []
    un_matched_orders_per_min = []
    matched_orders_per_min = []
    un_matched_orders = set()
    empty_vehicle_rate = []     # 车的空闲率
    print(method.__name__)

    for t in range(MIN_REQUEST_TIME, MAX_REQUEST_TIME):
        # vehicle_order_distance_per_min = []
        # vehicle_order_distance_per_min = {}
        # 将上一轮没有分配的订单和当前已经发布的订单合并(删除了超过时间的订单)
        un_matched_orders = update_orders(un_matched_orders, orders[t], t)
        un_matched_orders_per_min.append(len(un_matched_orders))
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
        print("service rate", len(matched_orders) / len(un_matched_orders))
        matched_orders_per_min.append(len(matched_orders))
        # 计算当前时间的服务率
        service_rate_result.append(len(matched_orders) / len(un_matched_orders))
        # 平台进行分配并将没有分配的订单存入一个
        un_matched_orders = un_matched_orders - matched_orders
        # # 计算每个时间点未匹配的订单和车之间的距离(只计算900-1440时间端的距离) 存储格式[剩余等待时间，车到订单起始点最小时间]
        # if t >= 899:
        #     for un_matched_order in un_matched_orders:
        #         # 计算每个单距离最近的车的距离
        #         min_distance = np.inf
        #         # 由两项组成 [剩余等待时间， 车到订单起始点的最小等待时间]
        #         time_pair = [un_matched_order.max_wait_time + un_matched_order.request_time - t]
        #         for vehicle in vehicles:
        #             distance_temp = shortest_distance[vehicle.location.osm_index][
        #                 un_matched_order.start_location.osm_index]
        #             if distance_temp < min_distance:
        #                 min_distance = distance_temp
        #         time_pair.append(min_distance / AVERAGE_SPEED)
        #         vehicle_order_distance_per_min[un_matched_order] = time_pair
        #         # vehicle_order_distance_per_min.append(time_pair)
        form_location = {}
        for vehicle in vehicles:
            form_location[vehicle] = vehicle.location.osm_index
        # 车辆更新位置
        update_vehicle_location(shortest_distance, shortest_path, shortest_path_with_minute, adjacent_nodes, vehicles, payments, t)
        for vehicle in vehicles:
            if form_location[vehicle] == vehicle.location.osm_index:
                print("位置没变", vehicle.vehicle_id, vehicle.location.osm_index, vehicle.goal_index, vehicle.route_plan)
        empty_vehicle_num = 0
        for vehicle in vehicles:
            # 统计空闲车的数量
            if vehicle.status == vehicle.WITHOUT_MISSION_STATUS:
                empty_vehicle_num += 1
        empty_vehicle_rate.append(empty_vehicle_num / len(vehicles))
        print(t)

    return social_welfare_result, social_cost_result, total_payment_result, total_utility_result, total_profit_result,\
        service_rate_result, un_matched_orders_per_min, matched_orders_per_min, empty_vehicle_rate


def order_dispatch_simulation(shortest_distance, shortest_path, shortest_path_with_minute, adjacent_nodes,
                              orders, vehicles, method):
    social_welfare_result = []
    social_cost_result = []
    total_profit_result = []
    total_payment_result = []
    total_utility_result = []
    service_rate_result = []
    un_matched_orders_per_min = []
    matched_orders_per_min = []
    empty_vehicle_rate_per_min = []  # 车的空闲率
    un_dispatched_orders = set()
    for t in range(MIN_REQUEST_TIME, MAX_REQUEST_TIME):
        # 将上一轮没有分配的订单和当前已经发布的订单合并
        un_dispatched_orders = update_orders(un_dispatched_orders, orders[t], t)
        un_matched_orders_per_min.append(len(un_dispatched_orders))
        # vehicle根据自己的情况进行投标并平台计算最优的分配并计算支付价格
        dispatch_result = method(shortest_distance, shortest_path, shortest_path_with_minute, adjacent_nodes, un_dispatched_orders, vehicles, t)
        # 记录结果
        dispatched_orders, payments, social_welfare, social_cost, total_payment, total_utility, total_profit, empty_vehicle_rate= dispatch_result
        social_welfare_result.append(social_welfare)
        social_cost_result.append(social_cost)
        total_payment_result.append(total_payment)
        total_utility_result.append(total_utility)
        total_profit_result.append(total_profit)
        print("service rate", len(dispatched_orders) / len(un_dispatched_orders))
        matched_orders_per_min.append(len(dispatched_orders))
        # 计算当前时间的服务率
        service_rate_result.append(len(dispatched_orders) / len(un_dispatched_orders))
        empty_vehicle_rate_per_min.append(empty_vehicle_rate)
        # 平台进行分配并将没有分配的订单存入一个
        un_dispatched_orders = un_dispatched_orders - dispatched_orders

        print(t)
    return social_welfare_result, social_cost_result, total_payment_result, total_utility_result, total_profit_result, \
        service_rate_result, un_matched_orders_per_min, matched_orders_per_min, empty_vehicle_rate


# SPARP
def order_dispatch_one_simulation(shortest_distance, shortest_path, shortest_path_with_minute, adjacent_nodes,
                              orders, vehicles):
    social_welfare_result = []
    social_cost_result = []
    total_profit_result = []
    total_payment_result = []
    total_utility_result = []
    empty_vehicle_rate = []  # 车的空闲率
    service_rate_result = []    # 每分钟服务率
    un_matched_orders_per_min = []  # 每分钟到来订单数目
    matched_orders_per_min = []  # 每分钟匹配订单数目

    # 还是按照一分钟的时间进行车辆位置的更新（一分钟以内车行进距离太短，很容易被修正到原来的位置）
    for t in range(MIN_REQUEST_TIME, MAX_REQUEST_TIME):
        print("round:", t)
        order_t = orders[t]
        served_order = 0
        social_welfare = 0.0
        social_cost = 0.0
        total_payment = 0.0
        total_utility = 0.0
        total_profit = 0.0
        un_matched_orders_per_min.append(len(order_t))
        for order in order_t:
            bids = {}
            costs = {}
            route_plans = {}
            rest_time = order.request_time + order.max_wait_time - t
            for vehicle in vehicles:    # 计算每个车的投标
                # 超过容积限制
                if vehicle.n_seats < order.n_riders:
                    bids[vehicle] = - np.inf
                    costs[vehicle] = - np.inf
                    route_plans[vehicle] = []
                    continue
                # 处在两个节点中间
                if vehicle.is_between == Vehicle.IS_BETWEEN_TWO_INDEX:
                    # 顺路或不能反向行驶
                    if shortest_distance[vehicle.goal_index, order.start_location.osm_index] + \
                        shortest_distance[vehicle.location.osm_index, vehicle.goal_index] - \
                        vehicle.between_distance < \
                        shortest_distance[vehicle.location.osm_index, order.start_location.osm_index] + \
                        vehicle.between_distance or \
                            shortest_distance[vehicle.goal_index, vehicle.location.osm_index] == np.inf:
                        if shortest_distance[vehicle.goal_index, order.start_location.osm_index] + \
                            shortest_distance[vehicle.location.osm_index, vehicle.goal_index] - \
                            vehicle.between_distance > \
                                Vehicle.AVERAGE_SPEED * rest_time:
                            bids[vehicle] = - np.inf
                            costs[vehicle] = - np.inf
                            route_plans[vehicle] = []
                            continue
                    else:
                        if shortest_distance[vehicle.location.osm_index, order.start_location.osm_index] + \
                            vehicle.between_distance > \
                                Vehicle.AVERAGE_SPEED * rest_time:
                            bids[vehicle] = - np.inf
                            costs[vehicle] = - np.inf
                            route_plans[vehicle] = []
                            continue
                else:
                    # 不在两个节点中间
                    if shortest_distance[vehicle.location.osm_index, order.start_location.osm_index] > \
                            Vehicle.AVERAGE_SPEED * rest_time:
                        bids[vehicle] = - np.inf
                        costs[vehicle] = - np.inf
                        route_plans[vehicle] = []
                        continue
                rem_list = vehicle.get_rem_list(vehicle.route_plan.copy())  # route_plan要转成只有起始点或单独一个终点的rem_list
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
                    served_order += 1
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
        form_location = {}
        for vehicle in vehicles:
            form_location[vehicle] = vehicle.location.osm_index
        #   一个t时间结束再进行车辆位置的更新
        for vehicle in vehicles:
            if vehicle.status == Vehicle.WITHOUT_MISSION_STATUS:
                vehicle.update_random_location(shortest_distance, shortest_path_with_minute, adjacent_nodes)
            else:
                vehicle.update_order_location(shortest_distance, shortest_path)
        for vehicle in vehicles:
            if form_location[vehicle] == vehicle.location.osm_index:
                print("位置没变", vehicle.vehicle_id, vehicle.location.osm_index, vehicle.goal_index, vehicle.route_plan)
        matched_orders_per_min.append(served_order)
        service_rate_result.append(served_order / len(order_t))
        print("service_rate:", served_order / len(order_t))
        social_welfare_result.append(social_welfare)
        social_cost_result.append(social_cost)
        total_payment_result.append(total_payment)
        total_utility_result.append(total_utility)
        total_profit_result.append(total_profit)
        # 统计空闲车的数量
        empty_vehicle_num = 0
        for vehicle in vehicles:
            if vehicle.status == vehicle.WITHOUT_MISSION_STATUS:
                empty_vehicle_num += 1
        empty_vehicle_rate.append(empty_vehicle_num / len(vehicles))
    return social_welfare_result, social_cost_result, total_payment_result, total_utility_result, total_profit_result, \
        service_rate_result, un_matched_orders_per_min, matched_orders_per_min, empty_vehicle_rate





if __name__ == '__main__':
    cur_model = "Multi_Nearest-Matching"

    start_time = time.clock()
    # with open("./env_data/static.pkl", "rb") as file:
    #     static = pickle.load(file)
    #     graph, shortest_distance, shortest_path, shortest_path_with_minute = static
    with open("./env_data/static.pkl", "rb") as file:
        static = pickle.load(file)
        graph, shortest_distance, shortest_path, shortest_path_with_minute, adjacent_nodes = static
    for k in range(0, 10):
        running_time_result = []
        print("repeat:", k)
        with open("./env_data/vehicle/{0}_{1}.pkl".format(VEHICLE_NUMBER, k), "rb") as file:
            vehicles = pickle.load(file)
            # # 座位全部改成三人
            # for vehicle in vehicles:
            #     vehicle.n_seats = 3
        with open("./env_data/order/{0}_{1}.pkl".format(VEHICLE_NUMBER, k), "rb") as file:
            orders = pickle.load(file)
        # # 复制一遍订单，变成两天的数据(要用copy.deepcopy())
        # new_orders = copy.deepcopy(orders)
        # for t in range(MIN_REQUEST_TIME, MAX_REQUEST_TIME):
        #     for order in new_orders[t]:
        #         order.request_time += 1440
        #     orders[t+1440] = new_orders[t]
        t1 = time.time()    # 计算算法耗时
        # 多单的情况
        env_data = shortest_distance, shortest_path, shortest_path_with_minute, adjacent_nodes, orders, vehicles, orders_dispatch_with_nearest_dispatch
        env_result = order_dispatch_simulation(*env_data)
        # env_data = shortest_distance, shortest_path, shortest_path_with_minute, adjacent_nodes, orders, vehicles, orders_matching_with_nearest_matching
        # env_result = order_matching_simulation(*env_data)
        # SPARP
        # env_data = shortest_distance, shortest_path, shortest_path_with_minute, adjacent_nodes, orders, vehicles
        # env_result = order_dispatch_one_simulation(*env_data)
        t2 = time.time()    # 计算算法耗时
        # 计算总服务率
        served_order_number, total_order_number = 0, 0
        for t in range(MIN_REQUEST_TIME, MAX_REQUEST_TIME):
            for order in orders[t]:
                if order.belong_to_vehicle:
                    served_order_number += 1
                total_order_number += 1
        # print("total_order", total_order_number)
        service_rate = served_order_number / total_order_number
        running_time_result.append(t2 - t1) # 计算算法耗时
        result = (*env_result, service_rate, running_time_result)
        # env_data = shortest_distance, shortest_path, shortest_path_with_minute, orders.copy(), vehicles
        # env_result = order_dispatch_one_simulation(*env_data)
        with open("./result/per_min/{0}/{1}_{2}.pkl".format(cur_model, VEHICLE_NUMBER, k), "wb") as file:
            pickle.dump(obj=result, file=file)

    elapsed = (time.clock() - start_time)
    print("time used:", elapsed)
