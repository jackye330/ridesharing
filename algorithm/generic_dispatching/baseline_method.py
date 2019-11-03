#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/26
import numpy as np
from typing import Set, List, Tuple, Dict
from agent.order import Order
from agent.vehicle import Vehicle

def orders_dispatch_with_nearest_dispatch(
        shortest_distance: np.ndarray, shortest_path: np.ndarray, shortest_path_with_minute: np.ndarray,
        adjacent_nodes: np.ndarray, orders: Set[Order], vehicles: List[Vehicle], current_time: int) \
        -> Tuple[Set[Order], Dict[Vehicle, List[Tuple[Order, float]]], float, float, float, float, float, float]:
    """
    按照最近车辆分配原则进行分配
    :param shortest_distance: 最短路径长度矩阵
    :param shortest_path: 最短路径矩阵
    :param shortest_path_with_minute: 最短路径矩阵下车辆下一分钟会到达的点
    :param adjacent_nodes: 相邻节点集合
    :param orders: 订单信息
    :param vehicles: 车辆集合
    :param current_time: 当前时间
    :return dispatched_orders: 已经匹配的订单 set: {Order}
    :return payments: 胜者支付 dict: {Vehicle:(Order,float)}
    :return social_welfare：社会福利
    :return social_cost: 胜者的总成本 float
    :return total_utility: 胜者的总效用 float
    :return total_profit: 平台的收益 float
    :return empty_vehicle_rate 一分钟后空车的比例
    """
    # 对于每一个订单取距离最近的车分配，如果这个车的调度是可以满足条件的，而且容量是可以满足需求的
    dispatched_orders = set()
    payments = {}
    social_welfare = 0.0
    social_cost = 0.0
    total_payment = 0.0
    total_utility = 0.0
    total_profit = 0.0
    print(type(orders))
    # 对于订单按照价值排序 ????
    orders = list(sorted(orders, key=lambda _order: _order.trip_fare))
    for order in orders:
        # 首先将车辆按照距离尽心排序(有的车在两个点中间)
        distance = {}
        for vehicle in vehicles:
            two_location_distance = shortest_distance[vehicle.location.osm_index, order.start_location.osm_index]
            if vehicle.is_between == Vehicle.IS_BETWEEN_TWO_INDEX:
                # 顺路或不能反向行驶
                if shortest_distance[vehicle.goal_index, order.start_location.osm_index] + \
                        shortest_distance[vehicle.location.osm_index, vehicle.goal_index] - vehicle.between_distance < \
                        two_location_distance + vehicle.between_distance or \
                        shortest_distance[vehicle.goal_index, vehicle.location.osm_index] == np.inf:
                    distance[vehicle] = shortest_distance[vehicle.goal_index, order.start_location.osm_index] + \
                                        shortest_distance[
                                            vehicle.location.osm_index, vehicle.goal_index] - vehicle.between_distance
                else:
                    distance[vehicle] = two_location_distance + vehicle.between_distance
            else:
                distance[vehicle] = shortest_distance[vehicle.location.osm_index][order.start_location.osm_index]

        vehicles.sort(key=lambda v: distance[v])

        for vehicle in vehicles:
            # 如果人数过多就继续考虑
            if order.n_riders > vehicle.n_seats:
                continue

            rest_of_time = order.max_wait_time + order.request_time - current_time

            # 如果当前的距离都不可以满足条件就不要指望后面更大的距离可以满足了
            if distance[vehicle] > rest_of_time * Vehicle.AVERAGE_SPEED:
                break

            # 计算已经添加这个订单之后的费用
            original_cost = vehicle.compute_cost(shortest_distance, vehicle.route_plan, current_time)
            current_cost, route_plan = vehicle.find_route_plan(shortest_distance, order, current_time)
            additional_cost = current_cost - original_cost

            if 0.0 < additional_cost < order.trip_fare:
                # 将订单分配给这个车辆
                dispatched_orders.add(order)
                vehicle.n_seats -= order.n_riders
                vehicle.route_plan = route_plan
                order.belong_to_vehicle = vehicle

                if vehicle not in payments:
                    payments[vehicle] = [(order, additional_cost)]
                else:
                    payments[vehicle].append((order, additional_cost))

                social_welfare += (order.trip_fare - additional_cost)
                social_cost += additional_cost
                total_payment += additional_cost
                total_utility += 0.0
                total_profit += (order.trip_fare - additional_cost)
                break

    vehicles.sort(key=lambda v: v.vehicle_id)
    form_location = {}
    for vehicle in vehicles:
        form_location[vehicle] = vehicle.location.osm_index
    # 车辆更新位置
    for vehicle in vehicles:
        # # 未激活车辆位置不变
        # if vehicle.activated == Vehicle.NEGATIVE:
        #     continue
        if vehicle not in payments:
            # 没有在胜者集合的而且没有路线目标的随机更新下一个位置
            if vehicle.status == Vehicle.WITHOUT_MISSION_STATUS:
                vehicle.update_random_location(shortest_distance, shortest_path_with_minute, adjacent_nodes)
            else:
                vehicle.update_order_location(shortest_distance, shortest_path)
            continue
        # 如果是胜者者的话就要决定最优路线而且还要更新订单的状态
        vehicle.update_order_location(shortest_distance, shortest_path)
    for vehicle in vehicles:
        if form_location[vehicle] == vehicle.location.osm_index:
            print("位置没变", vehicle.vehicle_id, vehicle.location.osm_index, vehicle.goal_index, vehicle.route_plan)
    empty_vehicle_num = 0
    for vehicle in vehicles:
        # 统计空闲车的数量
        if vehicle.status == vehicle.WITHOUT_MISSION_STATUS:
            empty_vehicle_num += 1
    empty_vehicle_rate = empty_vehicle_num / len(vehicles)
    return dispatched_orders, payments, social_welfare, social_cost, total_payment, total_utility, total_profit, empty_vehicle_rate