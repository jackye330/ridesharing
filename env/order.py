#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/22
from typing import Set

import numpy as np
import pandas as pd

from env.location import PickLocation, DropLocation
from env.network import Network
from setting import POINT_LENGTH
from preprocess.utility import RegionModel
from setting import FLOAT_ZERO, DETOUR_RATIOS, WAIT_TIMES, INT_ZERO

__all__ = ["Order", "generate_road_orders_data", "generate_grid_orders_data"]


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
        self._real_pick_up_time: int = INT_ZERO  # 车俩实际配分配的时间
        self._real_wait_time: int = INT_ZERO  # 实际等待时间
        self._real_service_time: int = INT_ZERO  # 车辆实际被服务的时间
        self._real_order_distance: float = FLOAT_ZERO  # 车辆被完成过程中多少距离是
        self._real_detour_ratio: float = FLOAT_ZERO  # 实际绕路比例

    @classmethod
    def set_order_generator(cls, generator):
        cls.order_generator = generator

    @classmethod
    def generate_orders_data(cls, output_file: str, network: Network):
        """
        将原始数据生成一个csv文件
        :param output_file: csv输出文件
        :param network: 网络
        :return:
        """
        cls.order_generator(output_file, network)

    @classmethod
    def load_orders_data(cls, start_time: int, time_slot: int, input_file: str):
        """
        从输入的csv文件中读取订单文件并逐个返回到外界
        :param start_time: 起始时间
        :param time_slot: 间隔时间
        :param input_file: csv输入文件
        :return:
        """
        chunk_size = 10000
        order_id = 0
        current_time = start_time
        each_time_slot_orders: Set[Order] = set()
        for csv_iterator in pd.read_table(input_file, chunksize=chunk_size, iterator=True):  # 这么弄主要是为了防止order_data过大
            for line in csv_iterator.values:
                # ["request_time", "wait_time", "pick_index", "drop_index", "order_distance", "order_fare", "detour_ratio", "n_riders"]
                each_order_data = line[0].split(',')
                request_time = int(each_order_data[0])
                wait_time = int(each_order_data[1])
                pick_index = int(each_order_data[2])
                drop_index = int(each_order_data[3])
                order_distance = float(each_order_data[4])
                order_fare = float(each_order_data[5])
                detour_ratio = float(each_order_data[6])
                n_riders = int(each_order_data[7])
                order = cls(
                    order_id=order_id,
                    pick_location=PickLocation(pick_index),
                    drop_location=DropLocation(drop_index),
                    request_time=request_time,
                    wait_time=wait_time,
                    order_distance=order_distance,
                    order_fare=order_fare,
                    detour_ratio=detour_ratio,
                    n_riders=n_riders,
                )
                if request_time < current_time + time_slot:
                    each_time_slot_orders.add(order)
                else:
                    current_time += time_slot
                    yield current_time, each_time_slot_orders
                    each_time_slot_orders.clear()
                    each_time_slot_orders.add(order)
                order_id += 1
        if len(each_time_slot_orders) != 0:
            yield current_time + time_slot, each_time_slot_orders

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


def generate_real_road_orders_data(output_file, *args, **kwargs):
    """
    实际路网环境中的订单生成和时间流逝
    :return:
    """
    # 这个是由真实的订单数据的生成需要的结果
    from setting import MIN_REQUEST_TIME, MAX_REQUEST_TIME
    shortest_distance = np.load("../data/Manhattan/network_data/shortest_distance.npy")
    order_data = pd.read_csv("../preprocess/raw_data/temp/Manhattan/order_data_{0:03d}.csv".format(0))
    order_data = order_data[(MIN_REQUEST_TIME <= order_data.pick_time) & (order_data.pick_time < MAX_REQUEST_TIME)]
    order_data = order_data[shortest_distance[order_data.pick_index, order_data.drop_index] != np.inf]
    order_data = order_data[shortest_distance[order_data.pick_index, order_data.drop_index] >= 1000.0]  # 过于短的或者订单的距离是无穷大
    order_data["wait_time"] = np.random.choice(WAIT_TIMES, size=order_data.shape[0])
    order_data["order_distance"] = shortest_distance[order_data.pick_index, order_data.drop_index]
    order_data["detour_ratio"] = np.random.choice(DETOUR_RATIOS, size=order_data.shape[0])
    order_data["n_riders"] = np.where(order_data.n_riders < 2, 1, 2)  # TODO 这一步是为了能保证2的上限, 以后可能需要修改
    order_data.drop(columns=["order_tip", "total_fare"], axis=1, inplace=True)
    order_data = order_data.rename(columns={'pick_time': 'request_time'})
    order_data["request_time"] = order_data["request_time"].values.astype(np.int32)
    order_data = order_data[["request_time", "wait_time", "pick_index", "drop_index", "order_distance", "order_fare", "detour_ratio", "n_riders"]]
    order_data.to_csv(output_file, index=False)


def generate_vir_road_orders_data(output_file: str, *args, **kwargs):
    # 这个是将30天的数据合并之后的结果
    from setting import MIN_REQUEST_TIME, MAX_REQUEST_TIME, ORDER_NUMBER_RATIO
    from setting import ORDER_DATA_FILES
    from setting import MILE_TO_M
    import pickle
    with open(ORDER_DATA_FILES["pick_region_model"], "rb") as file:
        pick_region_model: RegionModel = pickle.load(file)
    with open(ORDER_DATA_FILES["drop_region_model"], "rb") as file:
        drop_region_model: RegionModel = pickle.load(file)
    shortest_distance = np.load("../data/Manhattan/network_data/shortest_distance.npy")
    unit_fare_model = np.load(ORDER_DATA_FILES["unit_fare_model_file"])
    demand_model = np.load(ORDER_DATA_FILES["demand_model_file"])
    demand_location_model = np.load(ORDER_DATA_FILES["demand_location_model_file"])
    demand_transfer_model = np.load(ORDER_DATA_FILES["demand_transfer_model_file"])
    st_time_bin = MIN_REQUEST_TIME // 3600  # MIN_REQUEST_TIME 落在了哪一个时间区间上
    en_time_bin = (MAX_REQUEST_TIME - 1) // 3600  # MAX_REQUEST_TIME 落在了哪一个时间区间上
    # 第一个时间区域还需要生成的时间
    data_series = []
    for time_bin in range(st_time_bin, en_time_bin + 1):
        if st_time_bin == en_time_bin:  # 在同一个时间区间上
            demand_number = demand_model[time_bin] * (MAX_REQUEST_TIME - MIN_REQUEST_TIME) / 3600
        elif time_bin == st_time_bin:
            demand_number = demand_model[time_bin] * ((st_time_bin + 1) * 3600 - MIN_REQUEST_TIME) / 3600
        elif time_bin == en_time_bin:
            demand_number = demand_model[time_bin] * (MAX_REQUEST_TIME - en_time_bin * 3600) / 3600
        else:
            demand_number = demand_model[time_bin]
        demand_number = demand_number * ORDER_NUMBER_RATIO
        demand_prob_location = demand_location_model[time_bin]
        demand_prob_transfer = demand_transfer_model[time_bin]
        demand_number_of_each_transfer = np.zeros(shape=demand_prob_transfer.shape, dtype=np.int32)
        for i in range(demand_prob_transfer.shape[0]):
            demand_number_of_each_transfer[i] = np.round(demand_prob_location[i] * demand_prob_transfer[i] * demand_number, 0)
        for i in range(demand_number_of_each_transfer.shape[0]):
            for j in range(demand_number_of_each_transfer.shape[1]):
                d_n_of_t = demand_number_of_each_transfer[i, j]
                temp_order_data = pd.DataFrame()
                temp_order_data["request_time"] = np.random.randint(MIN_REQUEST_TIME, MAX_REQUEST_TIME, size=d_n_of_t)
                temp_order_data["wait_time"] = np.random.choice(WAIT_TIMES, size=d_n_of_t)
                temp_order_data["pick_index"] = np.array([pick_region_model.get_rand_index_by_region_id(i) for _ in range(d_n_of_t)], dtype=np.int16)
                temp_order_data["drop_index"] = np.array([drop_region_model.get_rand_index_by_region_id(j) for _ in range(d_n_of_t)], dtype=np.int16)
                temp_order_data["order_distance"] = shortest_distance[temp_order_data.pick_index.values, temp_order_data.drop_index.values]
                temp_order_data["order_fare"] = np.round(temp_order_data["order_distance"] * unit_fare_model[time_bin] / MILE_TO_M, POINT_LENGTH)
                temp_order_data["detour_ratio"] = np.random.choice(DETOUR_RATIOS, size=d_n_of_t)
                temp_order_data["n_riders"] = np.ones(shape=d_n_of_t, dtype=np.int8)
                temp_order_data = temp_order_data[(temp_order_data["order_distance"] != np.inf) & (temp_order_data["order_distance"] >= 1000.0)]
                temp_order_data = temp_order_data[["request_time", "wait_time", "pick_index", "drop_index", "order_distance", "order_fare", "detour_ratio", "n_riders"]]
                data_series.append(temp_order_data)
    order_data: pd.DataFrame = pd.concat(data_series, axis=0, ignore_index=True)
    order_data = order_data.sort_values(by="request_time", axis=0, ascending=True)
    order_data.to_csv(output_file, index=False)


def generate_road_orders_data(output_file: str, network: Network):
    """
    调用上面两类函数
    """
    generate_vir_road_orders_data(output_file, network)


def generate_grid_orders_data(output_file, network: Network):
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
    order_series = []
    for current_time in range(MIN_REQUEST_TIME, MAX_REQUEST_TIME):
        order_number = int(np.random.normal(MU, SIGMA))
        temp_order_data = pd.DataFrame()
        temp_order_data["request_time"] = np.ones(shape=(order_number,), dtype=np.int32) * current_time
        temp_order_data["wait_time"] = np.random.randint(MIN_WAIT_TIME, MAX_WAIT_TIME + 1, size=(order_number,))
        pick_locations = network.generate_random_locations(order_number, PickLocation)
        drop_locations = network.generate_random_locations(order_number, DropLocation)
        temp_order_data["pick_index"] = np.array([location.osm_index for location in pick_locations])
        temp_order_data["drop_index"] = np.array([location.osm_index for location in drop_locations])
        temp_order_data["order_distance"] = np.array([network.get_shortest_distance(pick_locations[idx], drop_locations[idx]) for idx in range(order_number)])
        temp_order_data["order_fare"] = temp_order_data.order_distance * UNIT_FARE
        temp_order_data["detour_ratio"] = np.random.choice(DETOUR_RATIOS, size=(order_number,))
        temp_order_data["n_riders"] = np.random.randint(MIN_N_RIDERS, MAX_N_RIDERS + 1, size=(order_number,))
        temp_order_data = temp_order_data[temp_order_data.pick_index != temp_order_data.drop_index]
        temp_order_data = temp_order_data[["request_time", "wait_time", "pick_index", "drop_index", "order_distance", "order_fare", "detour_ratio", "n_riders"]]
        order_series.append(temp_order_data)
    order_data = pd.concat(order_series, axis=0, ignore_index=True)
    order_data.to_csv(output_file, index=False)
