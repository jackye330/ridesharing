#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/14
from array import array
from queue import Queue
from typing import Set, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from agent.bid import OrderBid
from agent.vehicle import Vehicle
from env.order import Order


class BipartiteGraph:
    """
    由于vcg机制的二部图，这个只可以处理最大化社会福利问题
    """
    __slots__ = ["_sw", "_bids", "vehicle_index_match", "order_index_match",
                 "index2order", "order2index", "index2vehicle", "vehicle2index",
                 "deny_vehicle_index", "backup_sw_line"]

    def __init__(self, vehicle_set: Set[Vehicle], order_set: Set[Order]):
        # 内部与外面的对接
        self.index2vehicle = [vehicle for vehicle in vehicle_set]
        self.vehicle2index = {vehicle: i for i, vehicle in enumerate(self.index2vehicle)}
        self.index2order = [order for order in order_set]
        self.order2index = {order: i for i, order in enumerate(self.index2order)}
        vehicle_number = len(vehicle_set)
        order_number = len(order_set)
        self.vehicle_index_match = [array('h') for _ in range(vehicle_number)]
        self.order_index_match = [array('h') for _ in range(order_number)]
        self.deny_vehicle_index = -1
        self.backup_sw_line = None
        self._sw = np.zeros(shape=(vehicle_number, order_number))
        self._bids = None

    def set_bids(self, bids: Dict[Order, Dict[Vehicle, OrderBid]]):
        self._bids = bids

    def get_vehicle_order_pair_bid(self, vehicle: Vehicle, order: Order):
        return self._bids[order][vehicle]

    def add_edge(self, vehicle: Vehicle, order: Order, sw: float):
        vehicle_index = self.vehicle2index[vehicle]
        order_index = self.order2index[order]
        self.vehicle_index_match[vehicle_index].append(order_index)
        self.order_index_match[order_index].append(vehicle_index)
        self._sw[vehicle_index, order_index] = -sw   # linear_sum_assignment 只可以解决最小值问题，本问题是最大值问题所以这样处理

    def temp_remove_vehicle(self, vehicle: Vehicle):
        """
        暂时的剔除车辆
        :param vehicle:
        :return:
        """
        vehicle_index = self.vehicle2index[vehicle]
        self.deny_vehicle_index = vehicle_index   # 不允许匹配的车辆
        self.backup_sw_line = self._sw[vehicle_index, :].copy()  # 先备份后删除
        self._sw[vehicle_index, :] = 0

    def recovery_remove_vehicle(self):
        """
        修复剔除车辆带来的影响
        :return:
        """
        self._sw[self.deny_vehicle_index, :] = self.backup_sw_line
        self.backup_sw_line = None
        self.deny_vehicle_index = -1

    def get_sub_graph(self, st_order: Order, covered_order_index: List[bool], covered_vehicle_index: List[bool]):
        temp_order_set_ = set()
        temp_vehicle_set_ = set()
        bfs_queue = Queue()
        bfs_queue.put(st_order)
        temp_order_set_.add(st_order)

        st_order_index = self.order2index[st_order]
        covered_order_index[st_order_index] = True

        while not bfs_queue.empty():
            node = bfs_queue.get()
            if isinstance(node, Vehicle):
                vehicle_index = self.vehicle2index[node]
                for order_index in self.vehicle_index_match[vehicle_index]:
                    if not covered_order_index[order_index]:
                        covered_order_index[order_index] = True
                        order = self.index2order[order_index]
                        temp_order_set_.add(order)
                        bfs_queue.put(order)
            else:
                order_index = self.order2index[node]
                for vehicle_index in self.order_index_match[order_index]:
                    if not covered_vehicle_index[vehicle_index]:
                        covered_vehicle_index[vehicle_index] = True
                        vehicle = self.index2vehicle[vehicle_index]
                        temp_vehicle_set_.add(vehicle)
                        bfs_queue.put(vehicle)

        cls = type(self)
        sub_graph = cls(temp_vehicle_set_, temp_order_set_)

        for order in temp_order_set_:
            order_index = self.order2index[order]
            for vehicle_index in self.order_index_match[order_index]:
                vehicle = self.index2vehicle[vehicle_index]
                if vehicle not in temp_vehicle_set_:
                    continue
                sub_graph.add_edge(vehicle, order, -self._sw[vehicle_index, order_index])  # 这里取负号的原因是因为self._sw 里面都是负数值, 具体原因看add_edge函数的解释
        return sub_graph

    def get_sub_graphs(self):
        covered_order_index = [False] * len(self.index2order)       # 已经处理过的订单
        covered_vehicle_index = [False] * len(self.index2vehicle)   # 已经得到订单的车辆

        order_number = len(self.index2order)
        for order_index in range(order_number):
            if covered_order_index[order_index]:
                continue
            order = self.index2order[order_index]
            yield self.get_sub_graph(order, covered_order_index, covered_vehicle_index)  # 构建子图

    def maximum_weight_matching(self, return_match=False) -> Tuple[float, List[Tuple[Vehicle, Order]]]:
        """
        :param return_match: 如果返回匹配关系就要将订单和车辆的匹配对列表
        :return match: 订单和车辆的匹配对
        :return social_welfare: 社会福利
        """
        row_index, col_index = linear_sum_assignment(self._sw)
        social_welfare = -(self._sw[row_index, col_index].sum())
        match = []
        if return_match:
            for vehicle_index, order_index in zip(row_index, col_index):
                if vehicle_index != self.deny_vehicle_index:
                    vehicle = self.index2vehicle[vehicle_index]
                    order = self.index2order[order_index]
                    match.append((vehicle, order))
        return social_welfare, match


class MaximumWeightMatching:
    pass


class MarketClearingMatching:
    """
    利用市场清仓算法进行匹配工作，利用商品价格递增来去进行求解
    """
    __slots__ = ["epsilon"]

    def __init__(self):
        self.epsilon = 0.001

    def _check_is_prefect_matching(self):
        pass

    def maximum_weight_matching(self):
        pass