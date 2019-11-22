#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/25
from collections import defaultdict
from typing import List, Set, NoReturn

from agent.vehicle import Vehicle
from env.order import Order
from env.running_env import RideSharingEnv

__all__ = ["DispatchingMethod"]


class DispatchedResult:
    __slots__ = ["orders", "optimal_route", "reward"]

    def __init__(self):
        self.orders = []
        self.optimal_route = None
        self.reward = 0.0

    def add_order(self, order, reward):
        self.orders.append(order)
        self.reward += reward

    def set_route(self, route):
        self.optimal_route = route


class DispatchingMethod:
    """
    分配方法类
    dispatched_orders: 已经得到分配的订单
    dispatched_vehicles: 订单分发中获得订单的车辆集合
    dispatched_result: 分发结果, 包括车辆获得哪些订单和回报
    social_welfare：此轮分配的社会福利
    social_cost: 分配订单的车辆的运行成本
    total_driver_rewards: 分配订单车辆的总体支付
    total_driver_payoffs: 分配订单车辆的总效用和
    platform_profit: 平台在此轮运行中的收益
    """
    __slots__ = ["dispatched_orders", "dispatched_vehicles", "dispatched_results",
                 "social_welfare", "social_cost", "total_driver_rewards", "total_driver_payoffs", "platform_profit"]

    def __init__(self):
        self.dispatched_vehicles = set()
        self.dispatched_orders = set()
        self.dispatched_results = defaultdict(DispatchedResult)
        self.social_welfare = 0.0
        self.social_cost = 0.0
        self.total_driver_rewards = 0.0
        self.total_driver_payoffs = 0.0
        self.platform_profit = 0.0

    def reset(self):
        self.dispatched_vehicles.clear()
        self.dispatched_orders.clear()
        self.dispatched_results.clear()
        self.social_welfare = 0.0
        self.social_cost = 0.0
        self.total_driver_rewards = 0.0
        self.total_driver_payoffs = 0.0
        self.platform_profit = 0.0

    def run(self, vehicles: List[Vehicle], orders: Set[Order], env: RideSharingEnv) -> NoReturn:
        raise NotImplementedError
