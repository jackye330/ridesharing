#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/9
from typing import Set, NoReturn, List

from agent.vehicle import Vehicle
from algorithm.utility import Mechanism
from env.network import Network
from env.order import Order
from utility import singleton

__all__ = ["Platform"]


@singleton
class Platform:
    """
    平台
    dispatching_mechanism: 平台的运行的机制
    """
    __slots__ = ["_dispatching_mechanism", "_order_pool"]

    def __init__(self, dispatching_mechanism: Mechanism):
        self._order_pool: Set = set()
        self._dispatching_mechanism: Mechanism = dispatching_mechanism

    def collect_orders(self,  new_orders: Set[Order], current_time: int) -> NoReturn:
        """
        收集这一轮的新订单同时剔除一些已经过期的订单
        :param new_orders: 新的订单集合
        :param current_time: 当前时间
        :return:
        """
        unused_orders = set([order for order in self._order_pool if order.request_time + order.wait_time < current_time])  # 找到已经过期的订单
        self._order_pool -= unused_orders
        self._order_pool |= new_orders

    def remove_dispatched_orders(self) -> NoReturn:
        """
        从订单池子中移除已经得到分发的订单
        :return:
        """
        self._order_pool -= self._dispatching_mechanism.dispatched_orders

    def round_based_process(self, vehicles: List[Vehicle], new_orders: Set[Order], current_time: int, network: Network) -> NoReturn:
        """
        一轮运行过程
        :param vehicles: 车辆
        :param new_orders: 新产生的订单
        :param current_time:  当前时间
        :param network:  环境
        :return:
        """
        #  收集订单
        self.collect_orders(new_orders, current_time)

        # 匹配定价
        self._dispatching_mechanism.run(vehicles, self._order_pool, current_time, network)

        # 移除已经分配的订单
        self.remove_dispatched_orders()

    @property
    def order_pool(self) -> Set[Order]:
        return self._order_pool

    @property
    def dispatching_mechanism(self) -> Mechanism:
        return self._dispatching_mechanism
