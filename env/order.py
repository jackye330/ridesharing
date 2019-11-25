#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/22
from typing import Set

import numpy as np
import pandas as pd


from env.network import Network
from setting import SECOND_OF_DAY, FLOAT_ZERO, DISTANCE_EPS, DETOUR_RATIOS, WAIT_TIMES
from env.location import PickLocation, DropLocation
from utility import is_enough_small

__all__ = ["Order", "real_order_generator", "grid_order_generator"]


class FailReason:
    __slots__ = []
    OVER_WAIT_TIME = "OVER_WAIT_TIME"


class Order:
    """
    订单类
    order_id: 订单编号
    pick_location: 起始地
    drop_location: 终止地
    request_time: 请求时间
    wait_time:    等待时间
    order_distance: 订单行程距离
    order_fare: 订单费用
    detour_ratio: 最大可以容忍的绕路比
    n_riders: 订单的乘客数目
    """
    __slots__ = [
        "_order_id",
        "_pick_location",
        "_drop_location",
        "_request_time",
        "_wait_time",
        "_order_distance",
        "_order_fare",
        "_detour_distance",
        "_n_riders",
        "_pick_up_distance",
        "_belong_vehicle",
        "_have_finish",
        "_real_pick_up_time",
        "_real_order_distance",
        "_real_service_time",
        "_real_detour_ratio",
        "_real_wait_time",
    ]

    order_generator = None  # 设置订单生成器

    def __init__(self, order_id: int, pick_location: PickLocation, drop_location: DropLocation, request_time: int, wait_time: int, order_distance: float, order_fare: float, detour_ratio: float, n_riders: int):
        self._order_id: int = order_id
        self._pick_location: PickLocation = pick_location
        self._pick_location.set_belong_order(self)  # 反向设置，方便定位
        self._drop_location: DropLocation = drop_location
        self._drop_location.set_belong_order(self)  # 方向设置，方便定位
        self._request_time: int = request_time
        self._wait_time: int = wait_time
        self._order_distance: float = order_distance
        self._order_fare: float = order_fare
        self._detour_distance: float = detour_ratio * order_distance  # 最大绕路距离
        self._n_riders: int = n_riders
        self._pick_up_distance: float = FLOAT_ZERO  # 归属车辆为了接这一单已经行驶的距离
        self._belong_vehicle = None  # 订单归属车辆
        self._have_finish = False  # 订单已经被完成了
        self._real_pick_up_time: int = -1  # 车俩实际配分配的时间
        self._real_wait_time: int = -1   # 实际等待时间
        self._real_service_time: int = -1  # 车辆实际被服务的时间
        self._real_order_distance: float = FLOAT_ZERO  # 车辆被完成过程中多少距离是
        self._real_detour_ratio: float = FLOAT_ZERO  # 实际绕路比例

    @classmethod
    def set_order_generator(cls, generator):
        cls.order_generator = generator

    def __hash__(self):
        return hash(self._order_id)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise Exception("{0} is not {1}".format(other.__class__, self.__class__))
        return other._order_id == self._order_id

    def __repr__(self):
        return "(order_id: {0}, detour_ratio: {1})".format(self._order_id, self._detour_distance / self._order_distance)

    @property
    def order_id(self) -> int:
        return self._order_id

    @property
    def pick_location(self) -> PickLocation:
        return self._pick_location

    @property
    def drop_location(self) -> DropLocation:
        return self._drop_location

    @property
    def request_time(self) -> int:
        return self._request_time

    @property
    def wait_time(self) -> int:
        return self._wait_time

    @property
    def order_distance(self) -> float:
        return self._order_distance

    @property
    def order_fare(self) -> float:
        return self._order_fare

    @property
    def detour_distance(self) -> float:
        return self._detour_distance

    @property
    def pick_up_distance(self) -> float:
        return self._pick_up_distance

    @property
    def n_riders(self) -> int:
        return self._n_riders

    @property
    def belong_vehicle(self):
        return self._belong_vehicle

    @property
    def real_pick_up_time(self) -> int:
        return self._real_pick_up_time

    @property
    def real_order_distance(self) -> float:
        return self._real_order_distance

    @property
    def real_detour_ratio(self) -> float:
        return self._real_detour_ratio

    @property
    def real_wait_time(self) -> int:
        return self._real_wait_time

    @property
    def real_service_time(self) -> int:
        return self._real_service_time

    @property
    def turnaround_time(self) -> int:
        return self._real_wait_time + self._real_service_time

    def set_belong_vehicle(self, vehicle=None):
        self._belong_vehicle = vehicle

    def set_pick_status(self, pick_up_distance: float, real_pick_up_time: int):
        self._pick_up_distance = pick_up_distance
        self._real_pick_up_time = real_pick_up_time
        self._real_wait_time = real_pick_up_time - self._request_time

    def set_drop_status(self, drop_off_distance: float, real_finish_time: int):
        self._real_order_distance = drop_off_distance - self._pick_up_distance
        self._real_detour_ratio = self._real_order_distance / self._order_distance - 1.0
        self._real_service_time = real_finish_time - self._request_time - self._real_wait_time  # 这个订单被完成花费的时间


def real_order_generator(time_slot: int, network: Network):
    """
    实际路网环境中的订单生成和时间流逝
    :return:
    """
    # 订单的生成器
    from setting import MIN_REQUEST_TIME, MAX_REQUEST_TIME, MIN_REQUEST_DAY, MAX_REQUEST_DAY
    from setting import ORDER_DATA_FILES
    orders: Set[Order] = set()
    order_id = 0

    current_time = MIN_REQUEST_DAY * SECOND_OF_DAY + MIN_REQUEST_TIME
    for day in range(MIN_REQUEST_DAY, MAX_REQUEST_DAY):
        order_data = pd.read_csv(ORDER_DATA_FILES[day])
        order_data = order_data[MIN_REQUEST_TIME <= order_data["pick_time"]]
        order_data = order_data[MAX_REQUEST_TIME > order_data["pick_time"]]
        request_time_series = order_data["pick_time"].values
        pick_index_series = order_data["pick_index"].values
        drop_index_series = order_data["drop_index"].values
        n_riders_series = order_data["n_riders"].values
        order_fare_series = order_data["order_fare"].values  # 我们不考虑订单的中tip成分
        order_number = order_data.shape[0]
        detour_ratio_series = np.random.choice(DETOUR_RATIOS, size=(order_number,))

        for i in range(order_number):
            pick_location = PickLocation(int(pick_index_series[i]))
            drop_location = DropLocation(int(drop_index_series[i]))
            request_time = int(request_time_series[i]) + day * SECOND_OF_DAY
            wait_time = np.random.choice(WAIT_TIMES)
            order_distance = network.get_shortest_distance(pick_location, drop_location)
            if is_enough_small(order_distance, DISTANCE_EPS) or order_distance == np.inf:  # 过于短的或者订单的距离是无穷大
                continue
            order_fare = order_fare_series[i]
            detour_ratio = detour_ratio_series[i]
            n_riders = int(n_riders_series[i])
            order = Order(
                order_id=order_id,
                pick_location=pick_location, drop_location=drop_location,
                request_time=request_time, wait_time=wait_time, detour_ratio=detour_ratio,
                order_distance=order_distance, order_fare=order_fare, n_riders=n_riders)
            if request_time < current_time + time_slot:
                orders.add(order)
            else:
                current_time += time_slot
                yield current_time, orders
                orders.clear()
                orders.add(order)
            order_id += 1
        if len(orders) != 0:
            yield current_time + time_slot, orders


def grid_order_generator(time_slot: int, network: Network):
    """
    网格路网环境中的订单生成和时间流逝
    我们生成订单的方式可以参考论文 An Online Mechanism for Ridesharing in Autonomous Mobility-on-Demand Systems (IJCAI2016)
    """
    from setting import MIN_REQUEST_TIME, MAX_REQUEST_TIME
    from setting import MU, SIGMA
    from setting import MIN_WAIT_TIME, MAX_WAIT_TIME
    from setting import UNIT_FARE
    from setting import MIN_N_RIDERS, MAX_N_RIDERS
    from setting import DETOUR_RATIOS
    order_id = 0
    orders: Set[Order] = set()
    for current_time in range(MIN_REQUEST_TIME, MAX_REQUEST_TIME, time_slot):
        orders.clear()
        order_number = int(np.random.normal(MU, SIGMA))
        wait_times = np.random.randint(MIN_WAIT_TIME, MAX_WAIT_TIME + 1, size=(order_number,))
        pick_locations = network.generate_random_locations(order_number, PickLocation)
        drop_locations = network.generate_random_locations(order_number, DropLocation)
        order_distances = np.array([network.get_shortest_distance(pick_locations[idx], drop_locations[idx]) for idx in range(order_number)])
        order_fares = order_distances * UNIT_FARE
        detour_ratios = np.random.choice(DETOUR_RATIOS, size=(order_number,))
        n_riders = np.random.randint(MIN_N_RIDERS, MAX_N_RIDERS + 1, size=(order_number,))
        feasible_ides = [idx for idx in range(order_number) if not is_enough_small(order_distances[idx], FLOAT_ZERO)]
        yield current_time, set([
            Order(
                order_id=order_id + i, pick_location=pick_locations[idx], drop_location=drop_locations[idx],
                request_time=current_time, wait_time=wait_times[idx], detour_ratio=detour_ratios[idx],
                order_distance=order_distances[idx], order_fare=order_fares[idx],  n_riders=n_riders[idx]
            )
            for i, idx in enumerate(feasible_ides)
        ])
        order_id += len(feasible_ides)
