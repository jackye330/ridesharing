#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/7
from typing import List
from constant import INT_ZERO
from algorithm.route_planning.optimizer import PlanningOptimizer
from agent.vehicle import Vehicle
from agent.order import Order
from env.network import Network
from env.location import OrderLocation
from env.location import PickLocation
from env.location import DropLocation


class RoutePlanner:
    __slots__ = ["optimizer"]

    def __init__(self, optimizer: PlanningOptimizer):
        """
        :param optimizer: 路规划优化器
        """
        self.optimizer = optimizer

    def run(self, vehicle: Vehicle, order: Order, current_time: int):
        """
        :param vehicle: 车辆
        :param order: 订单
        :param current_time: 当前时间
        """
        raise NotImplementedError


class Inserting(RoutePlanner):
    """
    这个路径规划框架是 微软亚洲研究院 在 ICDE2013 中的论文里面采用的方法
    论文名称是 T-Share: A Large-Scale Dynamic Taxi Ridesharing Service T-Share
    该方法是保证原有的订单起始点的相对顺序不变，然后将新订单的起始位置和结束位置插入到原先的路径规划中。
    时间复杂度为 O(n^2*m) n为原先订单起始位置数目，m为进行优化的时间复度
    """
    __slots__ = []

    def __init__(self, optimizer: PlanningOptimizer):
        """
        :param network: 交通路网
        :param optimizer: 路径优化器可以争对每一个输入的路径规划优化内部的最优路径 选项MaximizeProfitOptimizer/MinimizeCostOptimizer
        """
        super().__init__(optimizer)

    def run(self, vehicle: Vehicle, order: Order, current_time: int):
        # 这里面所有操作都不可以改变自身原有的变量的值 ！！！！！
        self.optimizer.reset()  # 优化器初始化重要！！！！
        pick_location = order.pick_location  # 订单的起始点
        drop_location = order.drop_location  # 订单的终止点
        cur_route_plan = vehicle.route_plan  # 车辆此时的原始路径规划
        n = len(cur_route_plan)
        for i in range(n + 1):
            for j in range(i, n + 1):
                new_route_plan = cur_route_plan[:i] + [pick_location] + cur_route_plan[i:j] + [drop_location] + cur_route_plan[j:]
                self.optimizer.optimize(vehicle, new_route_plan, current_time)  # 优化器进行优化


class Rescheduling(RoutePlanner):
    """
    这个路径规划框架是 Mohammad Asghari 在 SIGSPATIAL-16 中论文里面采用的方法
    论文名称是 Price-aware Real-time Ride-sharing at Scale An Auction-based Approach
    该方法可以允许打乱原先的路径规划的的订单接送顺序，插入新订单的的起始位置。
    时间复杂度为 O(n!m) n为原先订单起始位置数目，m为进行优化的时间复度
    """
    def __init__(self, optimizer: PlanningOptimizer):
        super().__init__(optimizer)

    def run(self, vehicle: Vehicle, order: Order, current_time: int):
        # 这里面所有操作都不可以改变自身原有的变量的值 ！！！！！
        def _find_best_route_plan_with_recursion(_cur_list: List[OrderLocation], _rem_list: List[OrderLocation]):
            """
            递归函数，递归探索最优路劲
            :param _cur_list: 当前探索的节点
            :param _rem_list: 当前还没有探索的节点
            """
            if len(_rem_list) == INT_ZERO:
                self.optimizer.optimize(vehicle, _cur_list, current_time)  # 优化器进行优化
            else:
                _cur_list_copy = _cur_list.copy()
                for i, order_location in enumerate(_rem_list):
                    _cur_list_copy.append(order_location)
                    _rem_list_copy = _rem_list[:i] + _rem_list[i + 1:]
                    if isinstance(order_location, PickLocation):
                        _rem_list_copy.append(order_location.belong_to_order.drop_location)
                    _find_best_route_plan_with_recursion(_cur_list_copy, _rem_list_copy)
                    _cur_list_copy.pop()

        def _get_remain_list() -> List[OrderLocation]:
            """
            构造remain_list列表，只包含起始点或者单独的终点
            """
            _remain_list = []
            _remain_order_set = set()
            for order_location in vehicle.route_plan:
                if isinstance(order_location, PickLocation):  # order_location 是一个订单的起始点直接加入
                    _remain_list.append(order_location)
                    _remain_order_set.add(order_location.belong_to_order)
                else:  # order_location 是订单终点先判断 remain_list 里面有没有对应的起始点
                    if order_location.belong_to_order not in _remain_order_set:  # 如果这一单是只有起始点的就直接加入
                        _remain_list.append(order_location)
            _remain_list.append(order.pick_location)
            return _remain_list

        self.optimizer.reset()  # 优化器初始化！！！！
        _find_best_route_plan_with_recursion([], _get_remain_list())

