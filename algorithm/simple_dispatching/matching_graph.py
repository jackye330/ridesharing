#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/14
from collections import defaultdict
from queue import Queue
from typing import Set, List, Tuple
from setting import POINT_LENGTH
import numpy as np
from lapsolver import solve_dense  # 比 linear_sum_assignment 快10倍

from agent.vehicle import Vehicle
from setting import VALUE_EPS, INT_ZERO
from env.order import Order

__all__ = ["BipartiteGraph", "MaximumWeightMatchingGraph", "MarketClearingMatchingGraph"]


class BipartiteGraph:
    """
    由于vcg机制的二部图，这个只可以处理最大化社会福利问题
    """
    __slots__ = ["edge_num", "pair_social_welfare_matrix", "bids", "vehicle_link_orders", "order_link_vehicles", "vehicle_number", "order_number", "index2order", "order2index", "index2vehicle", "vehicle2index", "deny_vehicle_index", "backup_sw_line"]

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
        self.deny_vehicle_index = -1  # 初始化没有该禁止车辆
        self.bids = None
        self.edge_num = 0

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
        self.vehicle_number -= 1

    def recovery_remove_vehicle(self):
        """
        修复剔除车辆带来的影响
        :return:
        """
        self.deny_vehicle_index = -1
        self.vehicle_number += 1

    def add_edge(self, vehicle: Vehicle, order: Order, pair_social_welfare: float):
        self.vehicle_link_orders[vehicle].add(order)
        self.order_link_vehicles[order].add(vehicle)
        vehicle_index = self.vehicle2index[vehicle]
        order_index = self.order2index[order]
        self.pair_social_welfare_matrix[vehicle_index, order_index] = pair_social_welfare
        self.edge_num += 1

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
        if self.deny_vehicle_index != -1:
            sw_matrix = np.row_stack((self.pair_social_welfare_matrix[:self.deny_vehicle_index, :], self.pair_social_welfare_matrix[self.deny_vehicle_index+1:, :]))
        else:
            sw_matrix = self.pair_social_welfare_matrix.copy()

        if 0 in sw_matrix.shape:  # 如果存在一个维度不存在
            return 0, []

        row_index, col_index = solve_dense(-sw_matrix)  # linear_sum_assignment 只可以解决最小值问题，本问题是最大值问题所以这样处理
        social_welfare = np.round(sw_matrix[row_index, col_index].sum(), POINT_LENGTH)
        match_pairs = []
        if return_match:
            for vehicle_index, order_index in zip(row_index, col_index):
                if self.deny_vehicle_index != -1 and vehicle_index >= self.deny_vehicle_index:
                    vehicle_index += 1
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
        self.epsilon = 1 / pow(10, POINT_LENGTH)  # TODO 这递增值需要修改, 这个对于算法性能有很大的影响，这里暂时不使用这个算法

    @staticmethod
    def _is_prefect_matching(matches: List, m: int, n: int):
        m_o = np.ones(shape=(m,), dtype=np.int16) * -1
        m_v = np.ones(shape=(n,), dtype=np.int16) * -1
        v_v = np.zeros(shape=(n,), dtype=np.bool)

        def _greedy_matching() -> int:
            _cnt = INT_ZERO
            for i in range(m):
                for j in matches[i]:
                    if m_o[i] == -1 and m_v[j] == -1:
                        m_o[i], m_v[j] = j, i
                        _cnt += 1
                        break
            return _cnt

        def _find_augmenting_path(i: int) -> bool:
            for j in matches[i]:
                if v_v[j]:
                    continue
                v_v[j] = True
                if m_v[j] == -1 or _find_augmenting_path(m_v[j]):
                    m_o[i], m_v[j] = j, i
                    return True
            return False

        cnt = _greedy_matching()
        for _i in range(m):
            if m_o[_i] == -1:
                v_v = v_v * False
                if _find_augmenting_path(_i):
                    cnt += 1

        if cnt == m:
            return True, m_o
        else:
            return False, None

    @staticmethod
    def _get_match_result(payoff: np.ndarray):
        m, n = payoff.shape
        bid_cnt = np.zeros(n, dtype=np.int32)
        matches = [np.array([]) for _ in range(m)]
        for i in range(m):
            each_line = payoff[i, :]
            arg_max_idx = np.argwhere(np.abs(each_line - np.amax(each_line)) <= VALUE_EPS).flatten()
            matches[i] = arg_max_idx
            bid_cnt[arg_max_idx] += 1
        return bid_cnt, matches

    def maximal_weight_matching(self, return_match=False) -> Tuple[float, List[Tuple[Vehicle, Order]]]:
        if self.deny_vehicle_index != -1:
            sw_matrix = np.row_stack((self.pair_social_welfare_matrix[:self.deny_vehicle_index, :], self.pair_social_welfare_matrix[self.deny_vehicle_index+1:, :]))
        else:
            sw_matrix = self.pair_social_welfare_matrix.copy()

        if 0 in sw_matrix.shape:  # 如果存在一个维度不存在
            return 0, []

        if sw_matrix.shape[0] > sw_matrix.shape[1]:
            sw_matrix = sw_matrix.T
            transposed = True
        else:
            transposed = False

        m, n = sw_matrix.shape
        prices = np.zeros(shape=(n,))
        while True:
            payoff = np.round(sw_matrix - prices, 2)
            bid_cnt, matches = self._get_match_result(payoff)
            is_feasible, prefect_matching = self._is_prefect_matching(matches, m, n)
            if is_feasible:
                print(sw_matrix.shape[0], len(prefect_matching))
                social_welfare = np.round(sw_matrix[range(len(prefect_matching)), prefect_matching].sum(), POINT_LENGTH)
                match_pairs = []
                if return_match:
                    for i, j in enumerate(prefect_matching):
                        if transposed:
                            vehicle_index, order_index = j, i
                        else:
                            vehicle_index, order_index = i, j
                        if self.deny_vehicle_index != -1 and vehicle_index >= self.deny_vehicle_index:
                            vehicle_index += 1
                        vehicle = self.index2vehicle[vehicle_index]
                        order = self.index2order[order_index]
                        if order in self.vehicle_link_orders[vehicle]:
                            match_pairs.append((vehicle, order))
                return social_welfare, match_pairs
            else:
                prices[np.argwhere(bid_cnt > 1).flatten()] += self.epsilon
