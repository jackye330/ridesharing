#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/14
from array import array
from collections import defaultdict
from queue import Queue
from typing import Set, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from agent.vehicle import Vehicle
from setting import FLOAT_ZERO, VALUE_EPS
from env.order import Order

__all__ = ["BipartiteGraph", "MaximumWeightMatchingGraph", "MarketClearingMatchingGraph"]


class BipartiteGraph:
    """
    由于vcg机制的二部图，这个只可以处理最大化社会福利问题
    """
    __slots__ = ["pair_social_welfare_matrix", "bids", "vehicle_link_orders", "order_link_vehicles", "vehicle_number", "order_number", "index2order", "order2index", "index2vehicle", "vehicle2index", "deny_vehicle_index", "backup_sw_line"]

    def __init__(self, feasible_vehicles: Set[Vehicle], feasible_orders: Set[Order]):
        # 内部与外面的对接
        self.index2vehicle = [vehicle for vehicle in feasible_vehicles]
        self.vehicle2index = {vehicle: i for i, vehicle in enumerate(self.index2vehicle)}
        self.index2order = [order for order in feasible_orders]
        self.order2index = {order: i for i, order in enumerate(self.index2order)}
        self.vehicle_link_orders = defaultdict(set)
        self.order_link_vehicles = defaultdict(set)
        self.vehicle_number = len(feasible_vehicles)
        self.order_number = len(feasible_orders)
        self.pair_social_welfare_matrix = np.zeros(shape=(self.vehicle_number, self.order_number))
        self.deny_vehicle_index = -1
        self.backup_sw_line = None
        self.bids = None

    def get_vehicle_order_pair_bid(self, vehicle: Vehicle, order: Order):
        return self.bids[vehicle][order]

    def temporarily_remove_vehicle(self, vehicle: Vehicle):
        """
        暂时的剔除车辆
        :param vehicle:
        :return:
        """
        vehicle_index = self.vehicle2index[vehicle]
        self.deny_vehicle_index = vehicle_index  # 不允许匹配的车辆
        self.backup_sw_line = self.pair_social_welfare_matrix[vehicle_index, :].copy()  # 先备份后删除
        self.pair_social_welfare_matrix[vehicle_index, :] = FLOAT_ZERO

    def recovery_remove_vehicle(self):
        """
        修复剔除车辆带来的影响
        :return:
        """
        self.pair_social_welfare_matrix[self.deny_vehicle_index, :] = self.backup_sw_line
        self.backup_sw_line = None
        self.deny_vehicle_index = -1

    def add_edge(self, vehicle: Vehicle, order: Order, pair_social_welfare: float):
        self.vehicle_link_orders[vehicle].add(order)
        self.order_link_vehicles[order].add(vehicle)
        vehicle_index = self.vehicle2index[vehicle]
        order_index = self.order2index[order]
        self.pair_social_welfare_matrix[vehicle_index, order_index] = pair_social_welfare

    def get_sub_graph(self, st_order: Order, covered_vehicles: Set[Vehicle], covered_orders: Set[Order]):
        temp_vehicle_set = set()
        temp_order_set = set()
        bfs_queue = Queue()
        bfs_queue.put(st_order)
        temp_order_set.add(st_order)
        covered_orders.add(st_order)

        while not bfs_queue.empty():
            node = bfs_queue.get()
            if isinstance(node, Vehicle):
                for order in self.vehicle_link_orders[node]:
                    if order not in covered_orders:
                        covered_orders.add(order)
                        temp_order_set.add(order)
                        bfs_queue.put(order)
            else:
                for vehicle in self.order_link_vehicles[node]:
                    if vehicle not in covered_vehicles:
                        covered_vehicles.add(vehicle)
                        temp_vehicle_set.add(vehicle)
                        bfs_queue.put(vehicle)

        cls = type(self)
        sub_graph = cls(temp_vehicle_set, temp_order_set)
        for order in temp_order_set:
            for vehicle in self.order_link_vehicles[order]:
                if vehicle in temp_vehicle_set:
                    vehicle_index = self.vehicle2index[vehicle]
                    order_index = self.order2index[order]
                    sub_graph.add_edge(vehicle, order, self.pair_social_welfare_matrix[vehicle_index, order_index])
        sub_graph.bids = self.bids
        return sub_graph

    def get_sub_graphs(self):
        covered_vehicles = set()  # 已经覆盖了的车辆
        covered_orders = set()  # 已经覆盖了的订单

        for order in self.order2index:
            if order not in covered_orders:
                sub_graph = self.get_sub_graph(order, covered_vehicles, covered_orders)  # 构建子图
                yield sub_graph
                if sub_graph.order_number == self.order_number and sub_graph.vehicle_number == self.vehicle_number:  # 没有必要往后探索
                    break

    def maximal_weight_matching(self, return_match=False) -> Tuple[float, List[Tuple[Vehicle, Order]]]:
        """
        :param return_match: 如果返回匹配关系就要将订单和车辆的匹配对列表
        :return order_match: 订单和车辆的匹配对
        :return social_welfare: 社会福利
        """
        raise NotImplementedError


class MaximumWeightMatchingGraph(BipartiteGraph):
    __slots__ = []

    def __init__(self, feasible_vehicles: Set[Vehicle], feasible_orders: Set[Order]):
        super(MaximumWeightMatchingGraph, self).__init__(feasible_vehicles, feasible_orders)

    def maximal_weight_matching(self, return_match=False) -> Tuple[float, List[Tuple[Vehicle, Order]]]:
        row_index, col_index = linear_sum_assignment(-self.pair_social_welfare_matrix)  # linear_sum_assignment 只可以解决最小值问题，本问题是最大值问题所以这样处理
        social_welfare = self.pair_social_welfare_matrix[row_index, col_index].sum()
        match_pairs = []
        if return_match:
            for vehicle_index, order_index in zip(row_index, col_index):
                if vehicle_index != self.deny_vehicle_index:
                    vehicle = self.index2vehicle[vehicle_index]
                    order = self.index2order[order_index]
                    if order in self.vehicle_link_orders[vehicle]:
                        match_pairs.append((vehicle, order))
        return social_welfare, match_pairs


class MarketClearingMatchingGraph(BipartiteGraph):
    """
    利用市场清仓算法进行匹配工作，利用商品价格递增来去进行求解
    """
    __slots__ = ["epsilon"]

    def __init__(self, feasible_vehicles: Set[Vehicle], feasible_orders: Set[Order]):
        super(MarketClearingMatchingGraph, self).__init__(feasible_vehicles, feasible_orders)
        self.epsilon = 0.001

    def _is_prefect_matching(self, orders_matches: List[array]):
        m_v = np.ones(shape=(self.vehicle_number,), dtype=np.int16) * -1
        m_o = np.ones(shape=(self.order_number,), dtype=np.int16) * -1
        v_v = np.zeros(shape=(self.vehicle_number,), dtype=np.bool)

        def _greedy_matching() -> int:
            _cnt = 0
            for _order_index in range(self.order_number):
                if m_o[_order_index] == -1:
                    for _vehicle_index in orders_matches[_order_index]:
                        if m_v[_vehicle_index] == -1:
                            m_o[_order_index] = _vehicle_index
                            m_v[_vehicle_index] = _order_index
                            _cnt += 1
                            break
            return _cnt

        def _find_augmenting_path(_order_index: int) -> bool:
            for vehicle_index in orders_matches[_order_index]:
                if v_v[vehicle_index]:
                    continue
                v_v[vehicle_index] = True
                if m_v[vehicle_index] == -1 or _find_augmenting_path(m_v[vehicle_index]):
                    m_o[order_index] = vehicle_index
                    m_v[vehicle_index] = order_index
                    return True
            return False

        cnt = _greedy_matching()
        for order_index in range(self.order_number):
            if m_o[order_index] == -1:
                v_v = v_v * False
                if _find_augmenting_path(order_index):
                    cnt += 1

        if self.deny_vehicle_index == -1:
            if cnt == min(self.vehicle_number, self.order_number):
                return True, m_o
            else:
                return False, None
        else:
            if cnt == min(self.order_number, self.vehicle_number - 1):
                return True, m_o
            else:
                return False, None

    def _get_match_result(self, payoff: np.ndarray):
        vehicle_bid_cnt = defaultdict(int)  # 表示车辆被投标的次数
        orders_matches = [array('h') for _ in range(len(self.index2order))]
        for order_index in range(self.order_number):
            order_array = payoff[order_index, :]
            vehicle_indexes = np.argwhere(np.abs(order_array - np.amax(order_array)) <= VALUE_EPS).flatten()
            for vehicle_index in vehicle_indexes:
                if vehicle_index != self.deny_vehicle_index:
                    orders_matches[order_index].append(vehicle_index)
                    vehicle_bid_cnt[vehicle_index] += 1
        return vehicle_bid_cnt, orders_matches

    def maximal_weight_matching(self, return_match=False) -> Tuple[float, List[Tuple[Vehicle, Order]]]:
        prices = np.zeros(shape=(self.vehicle_number,))
        sw_matrix = self.pair_social_welfare_matrix.T
        while True:
            payoff = sw_matrix - prices
            vehicle_bid_cnt, orders_matches = self._get_match_result(payoff)
            is_feasible, prefect_matching = self._is_prefect_matching(orders_matches)
            if is_feasible:
                social_welfare = FLOAT_ZERO
                match_pairs = []
                if return_match:
                    for order_index, vehicle_index in enumerate(prefect_matching):
                        social_welfare += self.pair_social_welfare_matrix[vehicle_index, order_index]
                        vehicle = self.index2vehicle[vehicle_index]
                        order = self.index2order[order_index]
                        if order in self.vehicle_link_orders[vehicle]:
                            match_pairs.append((vehicle, order))
                return social_welfare, match_pairs
            else:
                for vehicle_index, cnt in vehicle_bid_cnt.items():
                    if cnt > 1:  # 这个车不仅被一个订单反向投标
                        prices[vehicle_index] += self.epsilon
