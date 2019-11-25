#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/8/4
from typing import List, Tuple, Set, Dict
import numpy as np

from agent import Order
from agent import Vehicle


def orders_dispatch_with_nearest_dispatch(
        shortest_distance: np.ndarray, shortest_path: np.ndarray, shortest_path_with_minute: np.ndarray,
        orders: Set[Order], vehicles: List[Vehicle], current_time: int) \
        -> Tuple[Set[Order], Dict[Vehicle, List[Tuple[Order, float]]], float, float, float, float, float]:
    """
    按照最近车辆分配原则进行分配
    :param shortest_distance: 最短路径长度矩阵
    :param shortest_path: 最短路径矩阵
    :param shortest_path_with_minute: 最短路径矩阵下车辆下一分钟会到达的点
    :param orders: 订单信息
    :param vehicles: 车辆集合
    :param current_time: 当前时间
    :return dispatched_orders: 已经匹配的订单 set: {Order}
    :return payments: 胜者支付 dict: {Vehicle:(Order,float)}
    :return social_welfare：社会福利
    :return social_cost: 胜者的总成本 float
    :return total_utility: 胜者的总效用 float
    :return total_profit: 平台的收益 float
    """
    # 对于每一个订单取距离最近的车分配，如果这个车的调度是可以满足条件的，而且容量是可以满足需求的
    dispatched_orders = set()
    payments = {}
    social_welfare = 0.0
    social_cost = 0.0
    total_payment = 0.0
    total_utility = 0.0
    total_profit = 0.0

    # 对于订单按照价值排序
    orders = list(sorted(orders, key=lambda _order: _order.trip_fare))
    for order in orders:
        # 首先将车辆按照距离尽心排序
        vehicles.sort(key=lambda v: shortest_distance[v.location.osm_index, order.start_location.osm_index])

        for vehicle in vehicles:
            two_location_distance = shortest_distance[vehicle.location.osm_index, order.start_location.osm_index]
            rest_of_time = order.max_wait_time + order.request_time - current_time

            # 如果当前的距离都不可以满足条件就不要指望后面更大的距离可以满足了
            if two_location_distance > rest_of_time * Vehicle.AVERAGE_SPEED:
                break

            # 如果人数过多就继续考虑
            if order.n_riders > vehicle.n_seats:
                continue

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
    for vehicle in vehicles:
        if vehicle not in payments:
            # 没有在胜者集合的而且没有路线目标的随机更新下一个位置
            if vehicle.status == Vehicle.WITHOUT_MISSION_STATUS:
                vehicle.update_random_location(shortest_distance, shortest_path_with_minute)
            else:
                vehicle.update_order_location(shortest_distance, shortest_path)
            continue

        vehicle.update_order_location(shortest_distance, shortest_path)

    return dispatched_orders, payments, social_welfare, social_cost, total_payment, total_utility, total_profit


def orders_dispatch_with_sequence_auction(
        shortest_distance: np.ndarray, shortest_path: np.ndarray, shortest_path_with_minute: np.ndarray,
        orders: Set[Order], vehicles: List[Vehicle], current_time: int) \
        -> Tuple[Set[Order], Dict[Vehicle, List[Tuple[Order, float]]], float, float, float, float, float]:
    # 对于每一个订单取距离最近的车分配，如果这个车的调度是可以满足条件的，而且容量是可以满足需求的
    dispatched_orders = set()
    payments = {}
    social_welfare = 0.0
    social_cost = 0.0
    total_payment = 0.0
    total_utility = 0.0
    total_profit = 0.0

    # 对于订单按照价值排序
    candidate_orders = list(sorted(orders, key=lambda _order: _order.trip_fare))

    for order in candidate_orders:

        candidate_vehicles_bids = list()

        for vehicle in vehicles:
            # 判断投标可行性 人数要满足条件 还有距离要可达
            two_location_distance = shortest_distance[vehicle.location.osm_index, order.start_location.osm_index]
            rest_of_time = order.max_wait_time + order.request_time - current_time
            if two_location_distance > rest_of_time * Vehicle.AVERAGE_SPEED or order.n_riders > vehicle.n_seats:
                continue

            original_cost = vehicle.compute_cost(shortest_distance, vehicle.route_plan, current_time)  # 计算没有投标的时候的费用
            current_cost, _ = vehicle.find_route_plan(shortest_distance, order, current_time)  # 计算已经添加这个订单之后的费用
            additional_cost = current_cost - original_cost  # 计算费用的增加量

            if 0 <= additional_cost <= order.trip_fare:
                candidate_vehicles_bids.append((vehicle, additional_cost))

        if len(candidate_vehicles_bids) == 0:
            pass
        else:
            dispatched_orders.add(order)
            candidate_vehicles_bids.sort(key=lambda bid: bid[1])
            winner_vehicle = candidate_vehicles_bids[0][0]
            winner_vehicle.n_seats -= order.n_riders
            _, route_plan = winner_vehicle.find_route_plan(shortest_distance, order, current_time)
            winner_vehicle.route_plan = route_plan
            order.belong_to_vehicle = winner_vehicle

            if len(candidate_vehicles_bids) == 1:
                payment = candidate_vehicles_bids[0][1]
            else:
                payment = candidate_vehicles_bids[1][1]  # 次价支付

            if winner_vehicle not in payments:
                payments[winner_vehicle] = [(order, payment)]
            else:
                payments[winner_vehicle].append((order, payment))

            social_welfare += (order.trip_fare - candidate_vehicles_bids[0][1])  # order.trip_fare - additional_cost
            social_cost += candidate_vehicles_bids[0][1]
            total_payment += payment
            total_utility += (payment - candidate_vehicles_bids[0][1])
            total_profit += (order.trip_fare - payment)

    vehicles.sort(key=lambda v: v.vehicle_id)
    for vehicle in vehicles:
        if vehicle not in payments:
            # 没有在胜者集合的而且没有路线目标的随机更新下一个位置
            if vehicle.status == Vehicle.WITHOUT_MISSION_STATUS:
                vehicle.update_random_location(shortest_distance, shortest_path_with_minute)
            else:
                vehicle.update_order_location(shortest_distance, shortest_path)
            continue

        vehicle.update_order_location(shortest_distance, shortest_path)

    return dispatched_orders, payments, social_welfare, social_cost, total_payment, total_utility, total_profit


if __name__ == '__main__':
    # TODO：测试algorithm的功能
    pass
