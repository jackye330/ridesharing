#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/22
from network.location import PickLocation
from network.location import DropLocation


class Order:
    __slots__ = ["order_id", "pick_location", "drop_location", "request_time", "max_wait_time", "order_distance",
                 "trip_fare", "detour_ratio", "n_riders", "pick_up_distance", "belong_to_vehicle"]

    def __init__(self, order_id: int, pick_location: PickLocation, drop_location: DropLocation,
                 request_time: int, max_wait_time: int, order_distance, trip_fare: float,
                 detour_ratio: float, n_riders: int):
        """
        构造函数
        :param order_id: 订单编号
        :param pick_location: 起始地
        :param drop_location: 终止地
        :param request_time: 请求时间
        :param max_wait_time: 最大等待时间
        :param order_distance: 订单行程距离
        :param trip_fare: 订单费用
        :param detour_ratio: 绕路比率
        :param n_riders: 订单的乘客
        """
        self.order_id = order_id
        self.pick_location = pick_location
        self.drop_location = drop_location
        self.request_time = request_time
        self.max_wait_time = max_wait_time
        self.order_distance = order_distance
        self.trip_fare = trip_fare
        self.detour_ratio = detour_ratio
        self.n_riders = n_riders
        self.pick_up_distance = 0.0    # 归属车辆接乘客已经行驶的里程
        self.belong_to_vehicle = None  # 订单归属车辆

    def set_vehicle_belong(self, vehicle=None):
        self.belong_to_vehicle = vehicle

    def __hash__(self):
        return hash(self.order_id)

    def __eq__(self, other):
        return other.order_id == self.order_id

    def __repr__(self):
        # return "{0} -> {1} fare:{2} detour_ratio:{3} rider_number:{4}". \
        #     format(self.start_location, self.end_location, self.trip_fare, self.detour_ratio, self.n_riders)
        return "start_time in {0} max_wait_time {1}, rider_number: {2}".\
            format(self.request_time, self.max_wait_time, self.n_riders)
