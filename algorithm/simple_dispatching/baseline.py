#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/26
from algorithm.utility import DispatchingResult
from agent.order import Order
from agent.vehicle import Vehicle
import numpy as np
from numpy.random import choice
from typing import List, Set, Tuple, Dict


def nearest_vehicle_matching(shortest_distance: np.ndarray, orders: Set[Order], vehicles: List[Vehicle], current_time: int) \
        -> DispatchingResult:
    """
    每一个订单按照最近的车辆匹配原则
    :param shortest_distance: 最短路径长度矩阵
    :param orders: 订单信息
    :param vehicles: 车辆集合
    :param current_time: 当前时间
    :return result: 匹配结果
    """
    # 对于每一个订单取距离最近的车分配，如果这个车的调度是可以满足条件的，而且容量是可以满足需求的
    result = DispatchingResult(
        matched_orders=set(),
        matched_vehicles=set(),
        payments={},
        social_welfare=0.0,
        social_cost=0.0,
        total_payment=0.0,
        total_utility=0.0,
        total_profit=0.0
    )

    for order in orders:
        # 争对当前订单各个车辆到达的距离
        distance = {}
        for vehicle in vehicles:
            distance[vehicle] = vehicle.compute_rest_pick_up_distance(order.start_location, shortest_distance)

        # 订单分配
        vehicles.sort(key=lambda v: distance[v])  # 首先将车辆按照距离进行排序(有的车辆在两个节点中间)
        for vehicle in vehicles:
            if order.n_riders > vehicle.n_seats:  # 如果人数过多就继续考虑
                continue

            if vehicle in result.matched_vehicles:  # 如果车辆已经被分配了
                continue

            rest_of_time = order.max_wait_time + order.request_time - current_time
            if distance[vehicle] > rest_of_time * Vehicle.AVERAGE_SPEED:  # 如果当前的距离都不可以满足条件就不要指望后面更大的距离可以满足了
                break

            # 计算已经添加这个订单之后的费用
            original_cost = vehicle.(shortest_distance, vehicle.route_plan, current_time)
            current_cost, route_plan = vehicle.find_route_plan(shortest_distance, order, current_time)
            additional_cost = current_cost - original_cost

            if 0.0 < additional_cost < order.trip_fare:
                result.payments[vehicle] = (order, additional_cost)
                result.social_welfare += (order.trip_fare - additional_cost)
                result.social_cost += additional_cost
                result.total_payment += additional_cost
                result.total_utility += 0.0
                result.total_profit += (order.trip_fare - additional_cost)
                matched_orders.add(order)
                matched_vehicles.add(vehicle)
                break

    vehicles.sort(key=lambda v: v.vehicle_id)
    result.matched_orders = matched_orders
    return result


def random_matching(shortest_distance: np.ndarray, orders: Set[Order], vehicles: List[Vehicle], current_time: int) \
        -> MatchResult:
    """
    按照最近车辆分配原则进行分配
    :param shortest_distance: 最短路径长度矩阵
    :param orders: 订单信息
    :param vehicles: 车辆集合
    :param current_time: 当前时间
    :return result: 匹配结果
    """
    # 对于每一个订单取距离最近的车分配，如果这个车的调度是可以满足条件的，而且容量是可以满足需求的
    matched_orders = set()
    matched_vehicles = set()
    result = MatchResult(
        matched_orders=set(),
        payments={},
        social_welfare=0.0,
        social_cost=0.0,
        total_payment=0.0,
        total_utility=0.0,
        total_profit=0.0
    )

    for order in orders:
        # 首先将车辆按照距离进行排序
        rest_of_time = order.max_wait_time + order.request_time - current_time

        tmp_vehicles_idx = [idx for idx in range(len(vehicles))
                            if vehicles[idx].n_seats >= order.n_riders and
                            vehicles[idx] not in matched_vehicles and
                            vehicles[idx].compute_rest_pick_up_distance(order.start_location, shortest_distance) <= rest_of_time * Vehicle.AVERAGE_SPEED]
        forbid = set()
        for _ in range(100):  # 只进行100轮查询 如果找不到一个随意车辆就放弃这一单
            if len(tmp_vehicles_idx) == 0:
                break
            idx = choice(tmp_vehicles_idx)
            if idx in forbid:
                continue

            forbid.add(idx)
            vehicle = vehicles[idx]
            original_cost = vehicle.compute_cost(shortest_distance, vehicle.route_plan, current_time)
            current_cost, route_plan = vehicle.find_route_plan(shortest_distance, order, current_time)
            additional_cost = current_cost - original_cost

            if 0.0 < additional_cost < order.trip_fare:
                # 将订单分配给这个车辆
                matched_orders.add(order)
                matched_vehicles.add(vehicle)
                vehicle.n_seats -= order.n_riders
                vehicle.route_plan = route_plan
                order.belong_to_vehicle = vehicle

                result.payments[vehicle] = (order, additional_cost)
                result.social_welfare += (order.trip_fare - additional_cost)
                result.social_cost += additional_cost
                result.total_payment += additional_cost
                result.total_utility += 0.0
                result.total_profit += (order.trip_fare - additional_cost)

                break


        return result

