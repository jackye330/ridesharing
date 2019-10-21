#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/2/18
import time
from queue import Queue
from typing import List, Tuple, Set, Dict
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from agent import Order
from agent import Vehicle


class BipartiteGraph:
    __slots__ = ["order_link_vehicle", "vehicle_link_order", "costs", "weights",
                 "index2order", "index2vehicle", "order2index", "vehicle2index", "w"]

    def __init__(self):
        self.order_link_vehicle = {}
        self.vehicle_link_order = {}
        self.costs = {}
        self.weights = {}
        self.index2order = {}
        self.index2vehicle = {}
        self.order2index = {}
        self.vehicle2index = {}
        self.w = []

    def add_edge(self, order: Order, vehicle: Vehicle, cost: float, weight: float):
        if order not in self.order_link_vehicle:
            self.order_link_vehicle[order] = set()
        self.order_link_vehicle[order].add(vehicle)

        if vehicle not in self.vehicle_link_order:
            self.vehicle_link_order[vehicle] = set()
        self.vehicle_link_order[vehicle].add(order)

        self.costs[order, vehicle] = cost
        self.weights[order, vehicle] = weight

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

    def remove_vehicle(self, vehicle: Vehicle):
        self.vehicle_link_order[vehicle] = []
        self.w[:, self.vehicle2index[vehicle]] = 0

    def add_vehicle(self, vehicle, vehicle_orders_set: Set[Vehicle]):
        self.vehicle_link_order[vehicle] = vehicle_orders_set
        vehicle_index = self.vehicle2index[vehicle]
        for order in vehicle_orders_set:
            self.w[self.order2index[order], vehicle_index] = -(self.weights[order, vehicle])

    def build_index(self):
        self.index2order = {i: order for i, order in enumerate(self.order_link_vehicle.keys())}
        self.order2index = {order: i for i, order in enumerate(self.order_link_vehicle.keys())}
        self.index2vehicle = {i: vehicle for i, vehicle in enumerate(self.vehicle_link_order.keys())}
        self.vehicle2index = {vehicle: i for i, vehicle in enumerate(self.vehicle_link_order.keys())}
        self.w = np.array([[0.0] * len(self.vehicle_link_order) for _ in range(len(self.order_link_vehicle))])
        for vehicle, link_orders in self.vehicle_link_order.items():
            vehicle_index = self.vehicle2index[vehicle]
            for order in link_orders:
                order_index = self.order2index[order]
                self.w[order_index, vehicle_index] = -(self.weights[order, vehicle])

    def maximum_weight_match(self, return_match=False):
        row_index, col_index = linear_sum_assignment(self.w)
        match = []
        if return_match:
            for order_index, vehicle_index in zip(row_index, col_index):
                order = self.index2order[order_index]
                vehicle = self.index2vehicle[vehicle_index]
                if order in self.vehicle_link_order[vehicle]:
                    match.append((order, vehicle))
        social_welfare = -(self.w[row_index, col_index].sum())
        return match, social_welfare


def update_orders(un_dispatched_orders: Set[Order], new_arise_orders: Set[Order], current_time: int) -> Set[Order]:
    """
    将上一步没有分配的订单和当前新产生的订单合并
    :param un_dispatched_orders: 上一时刻没有分配的订单
    :param new_arise_orders: 新产生的订单
    :param current_time: 当前时刻
    :return un_dispatched_orders_: 合并之后的结果 set: {order}
    """
    un_dispatched_orders_ = set()
    for order in un_dispatched_orders:
        if order.request_time + order.max_wait_time > current_time:
            un_dispatched_orders_.add(order)
    un_dispatched_orders_ |= new_arise_orders
    return un_dispatched_orders_


def dispatch_orders_with_vcg(row_size:int, col_size:int, orders: Set[Order], vehicles: List[Vehicle],
                             current_time: int, without_payment=False) \
        -> Tuple[Set[Order], Dict[Vehicle, Tuple[Order, float]], float, float, float, float, float]:
    """
    返回备选订单集合和备选车辆集合和投标
    根据当前的预备的车辆和订单消息以及投标中，计算分配关系和支付
    :param row_size: 网格大小
    :param col_size: 网格大小
    :param orders: 订单信息
    :param vehicles: 车辆集合
    :param current_time: 当前时间
    :return dispatched_orders: 已经匹配的订单 set: {Order}
    :return payments: 胜者支付 dict: {Vehicle:(Order,float)}
    :return social_welfare：社会福利
    :return social_cost: 胜者的总成本 float
    :return total_payment: 胜者的支付 float
    :return total_utility: 胜者的效用 float
    :return total_profit: 平台的收益 float
    """
    def __get_distance__(location1, location2):
        return  np.abs(location1.x - location2.x) + np.abs(location1.y - location2.y)
    # 汽车计算投标并构建二部图
    main_graph = BipartiteGraph()
    for vehicle in vehicles:
        original_cost = vehicle.compute_cost(vehicle.route_plan, current_time)  # 计算没有投标的时候的费用
        for order in orders:
            # 判断投标可行性 人数要满足条件 还有距离要可达
            two_location_distance = __get_distance__(vehicle.location, order.start_location)
            rest_of_time = order.max_wait_time + order.request_time - current_time
            if two_location_distance > rest_of_time * Vehicle.AVERAGE_SPEED or order.n_riders > vehicle.n_seats:
                continue

            current_cost, _ = vehicle.find_route_plan(order, current_time)  # 计算已经添加这个订单之后的费用
            additional_cost = current_cost - original_cost  # 计算费用的增加量

            if 0 <= additional_cost <= order.trip_fare:
                main_graph.add_edge(order, vehicle, additional_cost, order.trip_fare-additional_cost)

    dispatched_orders = set()
    payments = {}
    social_welfare = 0.0
    social_cost = 0.0
    total_payment = 0.0
    total_utility = 0.0
    total_profit = 0.0

    # SWMOM-VCG 拍卖
    check_order = set()    # 已经处理过的订单
    check_vehicle = set()  # 已经得到订单的车辆
    for order in main_graph.order_link_vehicle:
        if order in check_order:
            continue
        sub_graph = main_graph.get_sub_graph(order, check_order, check_vehicle)
        sub_graph.build_index()
        # WDP
        match, partial_social_welfare = sub_graph.maximum_weight_match(return_match=True)
        social_welfare += partial_social_welfare

        # VCG payment
        for each_match in match:
            without_order, without_vehicle = each_match
            trip_fare = without_order.trip_fare
            additional_cost = sub_graph.costs[without_order, without_vehicle]
            dispatched_orders.add(without_order)

            # 计算VCG价格
            if not without_payment:
                remove_vehicle_orders = sub_graph.vehicle_link_order[without_vehicle]
                sub_graph.remove_vehicle(without_vehicle)
                _, partial_social_welfare_without_vehicle = sub_graph.maximum_weight_match()
                sub_graph.add_vehicle(without_vehicle, remove_vehicle_orders)
                payment = additional_cost + (partial_social_welfare - partial_social_welfare_without_vehicle)
                payment = min(payment, trip_fare)
            else:
                payment = 0

            payments[without_vehicle] = (without_order, payment)
            social_cost += additional_cost
            total_payment += payment
            total_utility += (payment - additional_cost)
            total_profit += (trip_fare - payment)

    # 更新车辆
    for vehicle in vehicles:
        if vehicle not in payments:
            # 没有在胜者集合的而且没有路线目标的随机更新下一个位置
            if vehicle.status == Vehicle.WITHOUT_MISSION_STATUS:
                vehicle.update_random_location(row_size, col_size)
            else:
                vehicle.update_order_location()
            continue
        # 如果是胜者者的话就要决定最优路线而且还要更新订单的状态
        order, payment = payments[vehicle]
        _, best_route_plan = vehicle.find_route_plan(order, current_time)
        vehicle.route_plan = best_route_plan
        vehicle.n_seats -= order.n_riders
        order.belong_to_vehicle = vehicle
        vehicle.update_order_location()

    return dispatched_orders, payments, social_welfare, social_cost, total_payment, total_utility, total_profit


def dispatch_orders_with_lp(row_size:int, col_size:int,
        orders: Set[Order], vehicles: List[Vehicle], current_time:int) \
        -> Tuple[Set[Order], Dict[Vehicle, List[Tuple[Order, float]]], float, float, float, float, float]:
    """
    按照最近车辆分配原则进行分配
    :param row_size: 网格大小
    :param col_size: 网格大小
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
    pass



def dispatch_orders_with_nearest_vehicle(row_size: int, col_size: int,
        orders: Set[Order], vehicles: List[Vehicle], current_time: int, without_payment=False) \
        -> Tuple[Set[Order], Dict[Vehicle, List[Tuple[Order, float]]], float, float, float, float, float]:
    """
    按照最近车辆分配原则进行分配
    :param row_size: 网格大小
    :param col_size: 网格大小
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

    def __get_distance__(v_location, o_location):
        return np.abs(v_location.x - o_location.x) + np.abs(v_location.y - o_location.y)

    # 对于订单按照价值排序
    candidate_orders = list(sorted(orders, key=lambda _order: _order.trip_fare))
    have_order_vehicles = set()
    for order in candidate_orders:

        # 首先将车辆按照距离尽心排序
        vehicles.sort(key=lambda v: __get_distance__(v.location, order.start_location))

        for vehicle in vehicles:
            if vehicle in have_order_vehicles:
                # print("asdssada")
                continue
            two_location_distance = __get_distance__(vehicle.location, order.start_location)
            rest_of_time = order.max_wait_time + order.request_time - current_time

            # 如果当前的距离都不可以满足条件就不要指望后面更大的距离可以满足了
            if two_location_distance > rest_of_time * Vehicle.AVERAGE_SPEED:
                break

            # 如果人数过多就继续考虑
            if order.n_riders > vehicle.n_seats:
                continue

            # 计算已经添加这个订单之后的费用
            original_cost = vehicle.compute_cost(vehicle.route_plan, current_time)
            current_cost, route_plan = vehicle.find_route_plan(order, current_time)
            additional_cost = current_cost - original_cost

            if 0.0 < additional_cost < order.trip_fare:
                # 将订单分配给这个车辆
                dispatched_orders.add(order)
                have_order_vehicles.add(vehicle)
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
                vehicle.update_random_location(row_size, col_size)
            else:
                vehicle.update_order_location()
            continue

        vehicle.update_order_location()

    return dispatched_orders, payments, social_welfare, social_cost, total_payment, total_utility, total_profit


def dispatch_orders_with_greedy( row_size:int, col_size:int,
        orders: Set[Order], vehicles: List[Vehicle], current_time: int, without_payment=False) \
        -> Tuple[Set[Order], Dict[Vehicle, List[Tuple[Order, float]]], float, float, float, float, float]:
    # 对于每一个订单取距离最近的车分配，如果这个车的调度是可以满足条件的，而且容量是可以满足需求的
    dispatched_orders = set()
    payments = {}
    social_welfare = 0.0
    social_cost = 0.0
    total_payment = 0.0
    total_utility = 0.0
    total_profit = 0.0

    def __get_distance__(v_location, o_location):
        return np.abs(v_location.x - o_location.x) + np.abs(v_location.y - o_location.y)

    # 对于订单按照价值排序
    candidate_orders = list(sorted(orders, key=lambda _order: _order.trip_fare))

    for order in candidate_orders:

        candidate_vehicles_bids = list()

        for vehicle in vehicles:
            # 判断投标可行性 人数要满足条件 还有距离要可达
            two_location_distance = __get_distance__(vehicle.location, order.start_location)
            rest_of_time = order.max_wait_time + order.request_time - current_time
            if two_location_distance > rest_of_time * Vehicle.AVERAGE_SPEED or order.n_riders > vehicle.n_seats:
                continue

            original_cost = vehicle.compute_cost(vehicle.route_plan, current_time)  # 计算没有投标的时候的费用
            current_cost, _ = vehicle.find_route_plan(order, current_time)  # 计算已经添加这个订单之后的费用
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
            _, route_plan = winner_vehicle.find_route_plan(order, current_time)
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
                vehicle.update_random_location(row_size, col_size)
            else:
                vehicle.update_order_location()
            continue

        vehicle.update_order_location()

    return dispatched_orders, payments, social_welfare, social_cost, total_payment, total_utility, total_profit


def dispatch_orders_with_iterative(
        shortest_distance: np.ndarray, shortest_path: np.ndarray, shortest_path_with_minute: np.ndarray,
        orders: Set[Order], vehicles: List[Vehicle], current_time: int) \
        -> Tuple[Set[Order], Dict[Vehicle, Tuple[Order, float]], float, float, float, float, float]:
    # 汽车计算完成每一单的预期支付
    bids = {}
    candidate_orders = set()
    candidate_vehicles = set()
    for vehicle in vehicles:
        original_cost = vehicle.compute_cost(shortest_distance, vehicle.route_plan, current_time)  # 计算没有投标的时候的费用
        vehicle_bids = {}
        for order in orders:
            # 判断投标可行性 人数要满足条件 还有距离要可达
            two_location_distance = shortest_distance[vehicle.location.osm_index, order.start_location.osm_index]
            rest_of_time = order.max_wait_time + order.request_time - current_time
            if two_location_distance > rest_of_time * Vehicle.AVERAGE_SPEED or order.n_riders > vehicle.n_seats:
                continue

            current_cost, _ = vehicle.find_route_plan(shortest_distance, order, current_time)  # 计算已经添加这个订单之后的费用
            additional_cost = current_cost - original_cost  # 计算费用的增加量

            if 0 <= additional_cost <= order.trip_fare:
                candidate_orders.add(order)
                candidate_vehicles.add(vehicle)
                vehicle_bids[order] = additional_cost

        if len(vehicle_bids) != 0:  # 如果车辆的投标空间不是空的
            bids[vehicle] = vehicle_bids

    # 分配
    upper_price = pd.Series({order: order.trip_fare for order in candidate_orders})
    lower_price = pd.Series({order: 0.0 for order in candidate_orders})
    dispatched_orders = set()
    payments = {}
    social_welfare = 0.0
    social_cost = 0.0
    total_payment = 0.0
    total_utility = 0.0
    total_profit = 0.0

    while True:
        prices = (upper_price + lower_price) / 2
        match = {order: [] for order in candidate_orders}
        for vehicle in candidate_vehicles:
            target_order = None
            best_profit = 0.0
            for order, additional_cost in bids[vehicle].items():
                if order in dispatched_orders:
                    continue
                if best_profit < prices[order] - additional_cost:
                    best_profit = prices[order] - additional_cost
                    target_order = order

            if target_order:
                match[target_order].append(vehicle)

        for order in candidate_orders:
            if len(match[order]) == 1:
                vehicle = match[order][0]
                dispatched_orders.add(order)
                candidate_vehicles.remove(vehicle)
                payments[vehicle] = (order, prices[order])
                lower_price[order] = prices[order]
                upper_price[order] = prices[order]
                social_welfare += order.trip_fare - bids[vehicle][order]
                social_cost += bids[vehicle][order]
                total_payment += prices[order]
                total_utility += (prices[order] - bids[vehicle][order])
                total_profit += order.trip_fare - prices[order]
            elif len(match[order]) == 0:
                lower_price[order] = prices[order]
            else:
                upper_price[order] = prices[order]

        candidate_orders -= dispatched_orders  # 提出已经分配的订单
        if len(candidate_orders) == 0 or ((upper_price - lower_price) <= 0.01).all():  # 备选集合已经被选完就推出
            break

    # 更新车辆
    for vehicle in vehicles:
        if vehicle not in payments:
            # 没有在胜者集合的而且没有路线目标的随机更新下一个位置
            if vehicle.status == Vehicle.WITHOUT_MISSION_STATUS:
                vehicle.update_random_location(shortest_distance, shortest_path_with_minute)
            else:
                vehicle.update_order_location(shortest_distance, shortest_path)
            continue
        # 如果是胜者者的话就要决定最优路线而且还要更新订单的状态
        order, payment = payments[vehicle]
        _, best_route_plan = vehicle.find_route_plan(shortest_distance, order, current_time)
        vehicle.route_plan = best_route_plan
        vehicle.n_seats -= order.n_riders
        order.belong_to_vehicle = vehicle
        vehicle.update_order_location(shortest_distance, shortest_path)

    return dispatched_orders, payments, social_welfare, social_cost, total_payment, total_utility, total_profit


if __name__ == '__main__':
    # TODO：测试algorithm的功能
    pass
