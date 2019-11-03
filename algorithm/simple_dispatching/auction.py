#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/26
from queue import Queue
from typing import Tuple, Set, Dict, List
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from agent.order import Order
from agent.vehicle import Vehicle
from algorithm.orders_matching.utility import MatchResult


class BipartiteGraph:
    __slots__ = ["order_link_vehicle", "vehicle_link_order", "costs", "weights",
                 "index2order", "index2vehicle", "order2index", "vehicle2index", "sw"]

    def __init__(self):
        self.order_link_vehicle = {}
        self.vehicle_link_order = {}
        self.costs = {}  # 每一个匹配对的增加成本
        self.weights = {}  # 每一个匹配对的社会福利
        # 用于计算
        self.index2order = {}
        self.index2vehicle = {}
        self.order2index = {}
        self.vehicle2index = {}
        self.sw = np.array([])  # 用于求解最优匹配的矩阵

    def add_edge(self, order: Order, vehicle: Vehicle, cost: float, weight: float):
        if order not in self.order_link_vehicle:
            self.order_link_vehicle[order] = set()
        self.order_link_vehicle[order].add(vehicle)

        if vehicle not in self.vehicle_link_order:
            self.vehicle_link_order[vehicle] = set()
        self.vehicle_link_order[vehicle].add(order)

        self.costs[order, vehicle] = cost
        self.weights[order, vehicle] = weight

    def remove_vehicle(self, vehicle: Vehicle):
        self.vehicle_link_order[vehicle] = []
        self.sw[:, self.vehicle2index[vehicle]] = 0

    def add_vehicle(self, vehicle, vehicle_orders_set: Set[Vehicle]):
        self.vehicle_link_order[vehicle] = vehicle_orders_set
        vehicle_index = self.vehicle2index[vehicle]
        for order in vehicle_orders_set:
            self.sw[self.order2index[order], vehicle_index] = -(self.weights[order, vehicle])

    def build_index(self):
        self.index2order = {i: order for i, order in enumerate(self.order_link_vehicle.keys())}
        self.order2index = {order: i for i, order in enumerate(self.order_link_vehicle.keys())}
        self.index2vehicle = {i: vehicle for i, vehicle in enumerate(self.vehicle_link_order.keys())}
        self.vehicle2index = {vehicle: i for i, vehicle in enumerate(self.vehicle_link_order.keys())}
        self.sw = np.array([[0.0] * len(self.vehicle_link_order) for _ in range(len(self.order_link_vehicle))])
        for vehicle, link_orders in self.vehicle_link_order.items():
            vehicle_index = self.vehicle2index[vehicle]
            for order in link_orders:
                order_index = self.order2index[order]
                self.sw[order_index, vehicle_index] = -(self.weights[order, vehicle])

    def get_sub_graph(self, order: Order, check_order: Set[Order], check_vehicle: Set[Vehicle]):
        temp_order_set = set()
        temp_vehicle_set = set()

        temp_order_set.add(order)
        check_order.add(order)
        Q = Queue()
        Q.put(order)
        while not Q.empty():
            node = Q.get()
            if isinstance(node, Vehicle):
                for order in self.vehicle_link_order[node]:
                    if order not in check_order:
                        check_order.add(order)
                        temp_order_set.add(order)
                        Q.put(order)
            else:
                for vehicle in self.order_link_vehicle[node]:
                    if vehicle not in check_vehicle:
                        check_vehicle.add(vehicle)
                        temp_vehicle_set.add(vehicle)
                        Q.put(vehicle)

        cls = type(self)
        sub_graph = cls()
        for order in temp_order_set:
            for vehicle in self.order_link_vehicle[order]:
                if vehicle not in temp_vehicle_set:
                    continue
                sub_graph.add_edge(order, vehicle, self.costs[order, vehicle], self.weights[order, vehicle])
        return sub_graph

    def get_sub_graphs(self):
        check_order = set()  # 已经处理过的订单
        check_vehicle = set()  # 已经得到订单的车辆
        for order in self.order_link_vehicle:
            if order in check_order:
                continue

            # 构建子图
            sub_graph = self.get_sub_graph(order, check_order, check_vehicle)
            sub_graph.build_index()
            yield sub_graph

    def maximum_weight_match(self, return_match=False) -> Tuple[List[Tuple[Order, Vehicle]], float]:
        """
        :param return_match: 如果返回匹配关系就要将订单和车辆的匹配对列表
        :return match: 订单和车辆的匹配对
        :return social_welfare: 社会福利
        """
        row_index, col_index = linear_sum_assignment(self.sw)
        match = []
        if return_match:
            for order_index, vehicle_index in zip(row_index, col_index):
                order = self.index2order[order_index]
                vehicle = self.index2vehicle[vehicle_index]
                if order in self.vehicle_link_order[vehicle]:
                    match.append((order, vehicle))
        social_welfare = -(self.sw[row_index, col_index].sum())
        return match, social_welfare


def vcg_mechanism(bids: Dict[Vehicle, Dict[Order, float]]) \
        -> MatchResult:
    """
    使用二部图匹配决定分配，利用vcg价格进行支付，主要基vcg机制理论
    :param bids 司机投标
    :return result: 拍卖结果
    """
    # 初始化返回结果
    result = MatchResult(
        matched_orders=set(),
        matched_vehicles=set(),
        payments={},
        social_welfare=0.0,
        social_cost=0.0,
        total_payment=0.0,
        total_utility=0.0,
        total_profit=0.0
    )

    # 构建车辆与订单之间的二部匹配图
    main_graph = BipartiteGraph()
    for vehicle in bids:
        for order, additional_cost in bids[vehicle].items():
            main_graph.add_edge(order, vehicle, additional_cost, order.trip_fare - additional_cost)

    for sub_graph in main_graph.get_sub_graphs():
        # 胜者决定
        sub_match, sub_social_welfare = sub_graph.maximum_weight_match(return_match=True)
        result.social_welfare += sub_social_welfare

        # 定价计算
        for each_match_pair in sub_match:
            without_order, without_vehicle = each_match_pair

            # 计算VCG价格
            additional_cost = sub_graph.costs[without_order, without_vehicle]
            remove_vehicle_orders = sub_graph.vehicle_link_order[without_vehicle]
            sub_graph.remove_vehicle(without_vehicle)
            _, sub_social_welfare_without_vehicle = sub_graph.maximum_weight_match()
            sub_graph.add_vehicle(without_vehicle, remove_vehicle_orders)
            payment = additional_cost + (sub_social_welfare - sub_social_welfare_without_vehicle)
            payment = min(payment, without_order.trip_fare)

            # 保存结果
            result.payments[without_vehicle] = (without_order, payment)
            result.social_cost += additional_cost
            result.total_payment += payment
            result.total_utility += (payment - additional_cost)
            result.total_profit += (without_order.trip_fare - payment)
            result.matched_orders.add(without_order)
            result.matched_vehicles.add(without_vehicle)

    return result


def myerson_mechanism(bids: Dict[Vehicle, Dict[Order, float]]) \
        -> MatchResult:
    """
    使用贪心算法分配，使用临界价格进行支付，主要基于Myerson理论进行设计的机制
    :param bids: 司机投标
    :return result: 拍卖结果
    """
    result = MatchResult(
        matched_orders=set(),
        matched_vehicles=set(),
        payments={},
        social_welfare=0.0,
        social_cost=0.0,
        total_payment=0.0,
        total_utility=0.0,
        total_profit=0.0
    )

    # 构建资源池
    feasible_orders = set()
    feasible_vehicles = set()

    reverse_bids = defaultdict(set)  # 反向投标, 表示每一个订单投标车辆
    for without_vehicle, vehicle_bids in bids.items():
        for assigned_order, _ in vehicle_bids.items():
            reverse_bids[assigned_order].add(without_vehicle)
    pool = []
    for without_vehicle, vehicle_bids in bids.items():
        for assigned_order, additional_cost in vehicle_bids.items():
            if len(reverse_bids[assigned_order]) > 1:  # 只有当前的订单不止被一个车辆投标才会有放进资源池中, 这一招是为了阻止monopoly vehicle出现
                pool.append((assigned_order.trip_fare - additional_cost, assigned_order, without_vehicle))
                feasible_orders.add(assigned_order)
                feasible_vehicles.add(without_vehicle)
    pool.sort(key=lambda x: -x[0])
    del reverse_bids

    # 胜者决定
    for order_vehicle_pair in pool:
        if len(result.matched_orders) == len(feasible_orders) or len(result.matched_vehicles) == len(feasible_vehicles):
            break
        partial_social_welfare = order_vehicle_pair[0]
        assigned_order = order_vehicle_pair[1]
        without_vehicle = order_vehicle_pair[2]
        additional_cost = assigned_order.trip_fare - partial_social_welfare

        if assigned_order in result.matched_orders or without_vehicle in result.matched_vehicles:
            continue

        result.payments[without_vehicle] = [assigned_order, additional_cost]
        result.social_welfare += partial_social_welfare
        result.social_cost += additional_cost
        result.matched_orders.add(assigned_order)
        result.matched_vehicles.add(without_vehicle)

    # 定价计算
    for without_vehicle in result.matched_vehicles:
        matched_orders_ = set()
        matched_vehicles_ = set()
        feasible_vehicles.remove(without_vehicle)  # 移除了有什么用？把这个过程在走一遍

        assigned_order = result.payments[without_vehicle][0]
        additional_cost = result.payments[without_vehicle][1]
        payment = additional_cost
        for order_vehicle_pair in pool:
            if len(matched_orders_) == len(feasible_orders) or len(matched_vehicles_) == len(feasible_vehicles):
                break
            partial_social_welfare_ = order_vehicle_pair[0]
            order_ = order_vehicle_pair[1]
            vehicle_ = order_vehicle_pair[2]

            if vehicle_ in matched_vehicles_ or order_ in matched_orders_ or without_vehicle == vehicle_:
                continue
            payment = max(payment, assigned_order.trip_fare - partial_social_welfare_)
            matched_orders_.add(order_)
            matched_vehicles_.add(vehicle_)

            if assigned_order == order_:
                result.payments[without_vehicle] = (assigned_order, min(assigned_order.trip_fare, payment))
                break

        result.total_utility += (result.payments[without_vehicle][1] - additional_cost)
        result.total_payment += result.payments[without_vehicle][1]
        result.total_profit += (assigned_order.trip_fare - result.payments[without_vehicle][1])
        feasible_vehicles.add(without_vehicle)

    return result
