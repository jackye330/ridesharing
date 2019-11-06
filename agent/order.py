#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/22
from env.location import PickLocation
from env.location import DropLocation
from constant import FLOAT_ZERO

__all__ = ["Order"]


class Order:
    __slots__ = ["order_id", "pick_location", "drop_location", "request_time", "wait_time", "order_distance",
                 "order_fare", "detour_ratio", "n_riders", "pick_up_distance", "belong_to_vehicle"]

    def __init__(self, order_id: int, pick_location: PickLocation, drop_location: DropLocation,
                 request_time: int, wait_time: int, order_distance, order_fare: float,
                 detour_ratio: float, n_riders: int):
        """
        构造函数
        :param order_id: 订单编号
        :param pick_location: 起始地
        :param drop_location: 终止地
        :param request_time: 请求时间
        :param wait_time: 最大等待时间
        :param order_distance: 订单行程距离
        :param order_fare: 订单费用
        :param detour_ratio: 最大可以容忍的绕路比
        :param n_riders: 订单的乘客数目
        """
        self.order_id = order_id
        self.pick_location = pick_location
        self.drop_location = drop_location
        self.request_time = request_time
        self.wait_time = wait_time
        self.order_distance = order_distance
        self.order_fare = order_fare
        self.detour_ratio = detour_ratio
        self.n_riders = n_riders
        self.pick_up_distance = FLOAT_ZERO    # 归属车辆接乘客已经行驶的里程
        self.belong_to_vehicle = None         # 订单归属车辆

    @classmethod
    def generate_orders(cls):
        """
        这是一个订单生成器
        :return:
        """
        import pandas as pd
        from setting import MAX_REQUEST_DAY
        from setting import MAX_REQUEST_TIME
        from setting import MIN_REQUEST_DAY
        from setting import MIN_REQUEST_TIME
        from setting import ORDER_DATA_FILES

        order_id = 0
        for day in range(MIN_REQUEST_DAY, MAX_REQUEST_DAY):
            trip_order_data = pd.read_csv(ORDER_DATA_FILES[day])
            trip_order_data = trip_order_data[MIN_REQUEST_TIME <= trip_order_data["pick_time"]]
            trip_order_data = trip_order_data[MAX_REQUEST_TIME > trip_order_data["pick_time"]]
            pick_up_index_series = trip_order_data["pick_index"].values
            drop_off_index_series = trip_order_data["drop_index"].values
            request_time_series = trip_order_data["time"].values
            receive_fare_series = (trip_order_data["total_amount"] - trip_order_data["tip_amount"]).values  # 我们不考虑订单的中tip成分
            n_riders_series = trip_order_data["passenger_count"].values

            orders_on_each_day = {}
            for request_time in range(MIN_REQUEST_TIME, MAX_REQUEST_TIME):
                orders_on_each_day[request_time] = set()

            # order_id = 0
            # for i in range(order_number):
            #     order_id = i
            #     pick_location = PickLocation(int(pick_up_index_series[i]))
            #     drop_location = DropLocation(int(drop_off_index_series[i]))
            #     request_time = int(request_time_series[i])
            #     max_wait_time = np.random.choice(WAIT_TIMES)
            #     order_distance = shortest_distance[pick_location.osm_index, drop_location.osm_index]
            #     if order_distance == 0.0:
            #         continue
            #     receive_fare = receive_fare_series[i]
            #     detour_ratio = np.random.choice(DETOUR_RATIOS)
            #     n_riders = int(n_riders_series[i])
            #     order = Order(order_id, pick_location, drop_location, request_time, max_wait_time,
            #                   order_distance, receive_fare, detour_ratio, n_riders)
            #     # 订单数据
            #     pick_location.set_order_belong(order)
            #     drop_location.set_order_belong(order)
            #     orders_on_each_day[request_time].add(order)

    def set_vehicle_belong(self, vehicle=None):
        self.belong_to_vehicle = vehicle

    def __hash__(self):
        return hash(self.order_id)

    def __eq__(self, other):
        return other.order_id == self.order_id

    def __repr__(self):
        # return "{0} -> {1} fare:{2} detour_ratio:{3} rider_number:{4}". \
        #     format(self.start_location, self.end_location, self.order_fare, self.detour_ratio, self.n_riders)
        return "request_time in {0} wait_time {1}, rider_number: {2}".format(self.request_time, self.wait_time, self.n_riders)
