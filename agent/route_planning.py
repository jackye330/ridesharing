#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/26
import numpy as np
from inspect import getgeneratorstate
from typing import Dict
from typing import List
from constant import INF
from constant import INT_ZERO
from constant import FLOAT_ZERO
from agent.order import Order
from agent.vehicle import Vehicle
from env.location import OrderLocation
from env.location import PickLocation
from env.location import DropLocation
from env.network import Network
from utility import equal

__all__ = ["get_route_plan_info", "compute_profit", "compute_cost",
           "MaximizeProfitOptimizer", "MinimizeCostOptimizer", "inserting", "rescheduling"]


class RoutePlanInfo:
    """
    RoutePlanInfo: 一个车辆按照路线规划行走对应信息，包括路线规划的订单的绕路比，路线规划行驶的长度，路线规划是否可行
    """
    __slots__ = ["orders_detour_ratio", "route_plan_distance", "is_feasible"]

    def __init__(self, orders_detour_ratio: Dict[Order, float], route_plan_distance: float, is_feasible: bool):
        """
        :param orders_detour_ratio: 订单的绕路比
        :param route_plan_distance: 行驶路程长度
        :param is_feasible: 路劲规划的可行性
        """
        self.orders_detour_ratio = orders_detour_ratio
        self.route_plan_distance = route_plan_distance
        self.is_feasible = is_feasible


def get_route_plan_info(vehicle: Vehicle, new_route_plan: List[OrderLocation], current_time: int, network: Network)\
        -> RoutePlanInfo:
    """
    更具当前车的信息，求解一个新的路径规划的一些信息（每一个订单的绕路比，行驶长度，可行性）
    :param vehicle: 车辆
    :param new_route_plan: 新的的路线规划
    :param current_time: 当前时间
    :param network: 城市网格
    :return info: 路径规划信息
    """
    # 这里面所有操作都不可以改变车辆的实际的值，所有过程都是模拟，车辆实际还没有运行到这些点！！！！！！！
    if len(new_route_plan) == INT_ZERO:  # 如果是空的路径规划
        return RoutePlanInfo(orders_detour_ratio={}, route_plan_distance=FLOAT_ZERO, is_feasible=True)

    # if len([location for location in route_plan if location.order_location_type == OrderLocation.PICK_UP_TYPE]) > \
    #    len([location for location in route_plan if location.order_location_type == OrderLocation.DROP_OFF_TYPE]):
    #     # 如果是起点数目多于终点数目那么必然存在有些节点无法送到返回不可行
    #     return RoutePlanInfo(orders_detour_ratio={}, route_plan_distance=FLOAT_ZERO, is_feasible=False)

    current_drive_time = current_time
    current_drive_distance = vehicle.drive_distance
    # available_seats = vehicle.available_seats  # 如果动态判断上车可行性

    pick_up_distance = {}  # 记录每一个订单在被接到时行驶的距离
    orders_detour_ratio = {}  # 记录每一个订单的绕路比
    g = network.simulate_drive_on_route_plan(vehicle.location, new_route_plan)  # 初始化生成器
    for order_location, vehicle_order_distance in g:
        current_drive_time += (vehicle_order_distance / Vehicle.average_speed)  # 单位 s
        current_drive_distance += vehicle_order_distance  # 行驶距离
        belong_to_order = order_location.belong_to_order  # 订单坐标隶属的订单（这样赋值是浅复制？）
        if isinstance(order_location, PickLocation):
            # if available_seats - belong_to_order.n_riders <= INT_ZERO:  # 表示无法在此时加入新的乘客
            #     return RoutePlanInfo(orders_detour_ratio={}, route_plan_distance=FLOAT_ZERO, is_feasible=False)
            upper_time = belong_to_order.request_time + belong_to_order.wait_time
            if current_drive_time < upper_time or equal(current_drive_distance, upper_time):  # 无法满足最大等待时间
                break
            else:
                pick_up_distance[belong_to_order] = current_drive_distance  # 更新接乘客已经行驶的里程

        if isinstance(order_location, DropLocation):
            # available_seats += belong_to_order.n_riders
            if belong_to_order in pick_up_distance:
                real_order_distance = current_drive_distance - pick_up_distance[belong_to_order]
            else:
                if belong_to_order.pick_up_distance > current_drive_distance and \
                        not equal(belong_to_order.pick_up_distance, current_drive_distance):  # 路径规划中起始的和终止地判断是否合理
                    break
                real_order_distance = current_drive_distance - belong_to_order.pick_up_distance

            real_detour_ratio = (real_order_distance - belong_to_order.order_distance) / belong_to_order.order_distance  # 计算绕路比
            if real_detour_ratio > belong_to_order.detour_ratio and not equal(real_detour_ratio, belong_to_order.detour_ratio):  # 无法满足绕路比
                break
            else:
                orders_detour_ratio[belong_to_order] = real_detour_ratio

    if getgeneratorstate(g) == "GEN_CLOSED":  # 如果生成器正常退出
        original_drive_distance = vehicle.drive_distance  # 车辆之前已经行驶的距离
        info = RoutePlanInfo(  # TODO 找个时间将这个提出， 直接返回一个元组
            orders_detour_ratio=orders_detour_ratio,
            route_plan_distance=current_drive_distance - original_drive_distance,
            is_feasible=True)
    else:
        g.close()  # 关闭生成器
        info = RoutePlanInfo(orders_detour_ratio={}, route_plan_distance=INF, is_feasible=False)
    return info


def compute_cost(vehicle: Vehicle, new_route_plan: List[OrderLocation], current_time: int, network: Network) \
        -> float:
    """
    计算车辆的当前的位置按照某一个路线的成本
    :param vehicle: 车辆
    :param new_route_plan:  需要执行运算的路径规划，而非车辆自身的路径规划
    :param current_time: 当前时间
    :param network: 交通路网
    :return:
    """
    route_plan_info = get_route_plan_info(vehicle, new_route_plan, current_time, network)  # 获得路径规划的信息
    if route_plan_info.is_feasible:
        cost = vehicle.unit_cost * route_plan_info.route_plan_distance  # 计算成本
    else:
        cost = INF
    return cost


def compute_profit(vehicle: Vehicle, new_route_plan: List[OrderLocation], current_time: int, network: Network) \
        -> float:
    """
    按照当前路径规划下的对平台的收益
    :param vehicle: 车辆
    :param new_route_plan: 需要执行运算的路径规划，而非车辆自身的路径规划
    :param current_time: 当前时间
    :param network: 交通路网
    :return: profit: 利润
    """

    def _compute_discount_fare(order: Order, detour_ratio: float) -> float:
        """
        计算乘客打折费用, 为了激励乘客愿意接受绕路，给予费用上的打折优惠
        :param order: 订单
        :param detour_ratio: 订单的绕路比
        :return:
        """
        return order.order_fare * (1 - 0.25 * detour_ratio * detour_ratio)

    route_plan_info = get_route_plan_info(vehicle, new_route_plan, current_time, network)
    if route_plan_info.is_feasible:
        fare = sum([_compute_discount_fare(order, detour_ratio)
                    for order, detour_ratio in route_plan_info.orders_detour_ratio.items()])
        cost = vehicle.unit_cost * route_plan_info.route_plan_distance
        profit = fare - cost
    else:
        profit = -INF

    return profit


class PlanningOptimizer:
    __slots__ = ["best_value", "best_route_plan", "compute_value_method"]

    def __init__(self, best_value: float, compute_value_method):
        self.best_value = best_value
        self.best_route_plan = []
        self.compute_value_method = compute_value_method

    def reset(self):
        raise NotImplementedError

    def optimize(self, vehicle: Vehicle, new_route_plan: List[OrderLocation], current_time: int, network: Network):
        raise NotImplementedError

    def get_optimize_value(self):
        return self.best_value

    def get_best_route_plan(self):
        return self.best_route_plan


class MaximizeProfitOptimizer(PlanningOptimizer):
    __slots__ = []

    def __init__(self, profit=-np.inf):
        super().__init__(profit, compute_profit)  # 这里面profit是要去进行优化的目标

    def reset(self):
        self.best_value = -INF  # 由于需要最大化利润，所以初始化为无穷小
        self.best_route_plan = []

    def optimize(self, vehicle: Vehicle, new_route_plan: List[OrderLocation], current_time: int, network: Network):
        profit = self.compute_value_method(vehicle, new_route_plan, current_time, network)
        if self.best_value < profit:
            self.best_value = profit
            self.best_route_plan = new_route_plan


class MinimizeCostOptimizer(PlanningOptimizer):
    __slots__ = []

    def __init__(self, cost=np.inf):
        super().__init__(cost, compute_cost)  # 这里面cost是要去进行优化的目标

    def reset(self):
        self.best_value = INF  # 由于需要最小化成本，所以初始化为无穷大
        self.best_route_plan = []

    def optimize(self, vehicle: Vehicle, new_route_plan: List[OrderLocation], current_time: int, network: Network):
        cost = self.compute_value_method(vehicle, new_route_plan, current_time, network)
        if self.best_value > cost:
            self.best_value = cost
            self.best_route_plan = new_route_plan


def inserting(vehicle: Vehicle, order: Order, current_time: int, network: Network, optimizer: PlanningOptimizer):
    """
    这个路径规划框架是 微软亚洲研究院 在 ICDE2013 中的论文里面采用的方法
    论文名称是 T-Share: A Large-Scale Dynamic Taxi Ridesharing Service T-Share
    该方法是保证原有的订单起始点的相对顺序不变，然后将新订单的起始位置和结束位置插入到原先的路径规划中。
    时间复杂度为 O(n^2*m) n为原先订单起始位置数目，m为进行优化的时间复度

    :param vehicle: 车辆
    :param order: 订单
    :param current_time: 当前时间
    :param network: 交通路网
    :param optimizer: 路径优化器可以争对每一个输入的路径规划优化内部的最优路径 选项MaximizeProfitOptimizer/MinimizeCostOptimizer
    """
    # 这里面所有操作都不可以改变自身原有的变量的值 ！！！！！
    optimizer.reset()    # 优化器初始化重要！！！！
    pick_location = order.pick_location  # 订单的起始点
    drop_location = order.drop_location  # 订单的终止点
    cur_route_plan = vehicle.route_plan  # 车辆此时的原始路径规划
    n = len(cur_route_plan)
    for i in range(n + 1):
        for j in range(i, n + 1):
            new_route_plan = cur_route_plan[:i] + [pick_location] + cur_route_plan[i:j] + [drop_location] + cur_route_plan[j:]
            optimizer.optimize(vehicle, new_route_plan, current_time, network)  # 优化器进行优化


def rescheduling(vehicle: Vehicle, order: Order, current_time: int, network: Network, optimizer: PlanningOptimizer):
    """
    这个路径规划框架是 Mohammad Asghari 在 SIGSPATIAL-16 中论文里面采用的方法
    论文名称是 Price-aware Real-time Ride-sharing at Scale An Auction-based Approach
    该方法可以允许打乱原先的路径规划的的订单接送顺序，插入新订单的的起始位置。
    时间复杂度为 O(n!m) n为原先订单起始位置数目，m为进行优化的时间复度
    :param vehicle: 车辆
    :param order: 订单
    :param current_time: 当前时间
    :param network: 交通路网
    :param optimizer: 路径优化器可以争对每一个输入的路径规划优化内部的最优路径 选项MaximizeProfitOptimizer/MinimizeCostOptimizer
    """
    # 这里面所有操作都不可以改变自身原有的变量的值 ！！！！！
    def _find_best_route_plan_with_recursion(_cur_list: List[OrderLocation], _rem_list: List[OrderLocation]):
        """
        递归函数，递归探索最优路劲
        :param _cur_list: 当前探索的节点
        :param _rem_list: 当前还没有探索的节点
        """
        if len(_rem_list) == INT_ZERO:
            optimizer.optimize(vehicle, _cur_list, current_time, network)  # 优化器进行优化
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

    optimizer.reset()  # 优化器初始化！！！！
    _find_best_route_plan_with_recursion([], _get_remain_list())
