#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/22
from typing import List

import numpy as np

from agent.order import Order
from network.transport_graph import _Graph
from network.transport_graph1.location import VehicleLocation, PickLocation, DropLocation
from setting import AVERAGE_SPEED
from setting import TIME_SLOT

__all__ = ["Vehicle"]


class Vehicle:
    __slots__ = ["vehicle_id", "location", "n_seats", "unit_cost",
                 "route_plan", "have_mission", "drive_distance", "is_activated"]
    average_speed = 0.0  # 车辆平均速度
    bidding_strategy = None  # 车辆投标策略

    def __init__(self, vehicle_id: int, location: VehicleLocation, n_seats: int, unit_cost: float):
        """
        构造函数
        :param vehicle_id: 车辆id
        :param location: 车辆当前位置
        :param n_seats:  车辆剩下的位置数目
        :param unit_cost: 车俩每一公里的成本
        """
        self.vehicle_id = vehicle_id  # 车辆唯一标识
        self.location = location  # 车辆当前的位置
        self.n_seats = n_seats  # 剩下的座位数目
        self.unit_cost = unit_cost  # 单位行驶成本
        self.route_plan = []  # 路径规划
        self.have_mission = False  # 默认车辆一开始是没有接送任务的
        self.is_activated = True  # 默认车都是激活状态
        self.drive_distance = 0.0  # 车辆总行驶距离

    @classmethod
    def set_average_speed(cls, average_speed):
        cls.average_speed = average_speed

    @classmethod
    def set_bidding_strategy(cls, strategy):
        cls.bidding_strategy = strategy

    def compute_bids(self, orders: List[Order], shortest_distance: np.ndarray, current_time: int):
        cls = type(self)
        bidding_strategy = cls.bidding_strategy
        bidding_strategy.get_bids(self, orders, shortest_distance, current_time)

    def drive_on_random(self, network: _Graph):  # TODO 需要修改
        """
        车辆随机在路上行驶
        :param network: 交通路网
        :return:
        ------
        注意：
        不要那些只可以进去，不可以出来的节点
        如果车辆就正好在一个节点之上，那么随机选择一个节点到达，如果不是这些情况就在原地保持不动
        """
        could_drive_distance = AVERAGE_SPEED * TIME_SLOT
        self.drive_distance += network.real_vehicle_drive_on_random(self.location, could_drive_distance)

    def drive_on_route_plan(self, network: _Graph):
        """
        车辆自己按照自己的路径规划行驶
        :param network: 交通路网
        """
        un_covered_location_index = 0  # 未完成订单坐标的最小索引
        could_drive_distance = AVERAGE_SPEED * TIME_SLOT  # 当前时间间隙内需要行驶的距离

        g = network.real_vehicle_drive_on_route_plan(self.location, self.route_plan, could_drive_distance)
        for is_access, covered_index, order_location, vehicle_to_order_distance in g:
            # is_access 表示是否可以到达 order_location
            # covered_index 表示车辆当前覆盖的路线规划列表索引
            # order_location 表示当前可以访问的订单位置
            # partial_drive_distance 表示车辆到 order_location 可以行驶的距离

            self.drive_distance += vehicle_to_order_distance
            un_covered_location_index = covered_index + 1  # 更新未完成订单的情况
            if is_access:  # 如果当前订单是可以到达的情况
                belong_to_order = order_location.belong_to_order
                if isinstance(order_location, PickLocation):
                    belong_to_order.pick_up_distance = self.drive_distance  # 更新当前订单的接送行驶距离
                    # self.n_seats -= belong_to_order.n_riders  # 这种更新适用与动态更新车辆可用座位数量
                if isinstance(order_location, DropLocation):
                    self.n_seats += belong_to_order.n_riders  # 因为送到乘客了，可以腾出位置

        self.route_plan = self.route_plan[un_covered_location_index:]  # 更新自己的路线规划
        if len(self.route_plan) != 0:
            self.have_mission = True
        else:
            self.have_mission = False

    def __repr__(self):
        return "id is {0} location = {1} available_seats = {2} trip_distance = {3} route_plan = {4}". \
            format(self.vehicle_id, self.location, self.n_seats, self.drive_distance, self.route_plan)

    def __hash__(self):
        return hash(self.vehicle_id)

    def __eq__(self, other):
        return self.vehicle_id == other.vehicle_id
