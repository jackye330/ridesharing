#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/3/3
from typing import List, Tuple

import numpy as np


class GeoLocation:
    __slots__ = ["x", "y", "osmid"]

    def __init__(self, osmid: int, x: int, y: int):
        self.osmid = osmid
        self.x = x
        self.y = y

    def __repr__(self):
        return "({0},{1})".format(self.x, self.y)

    def __hash__(self):
        return hash(self.osmid)

    def __eq__(self, other):
        return other.osmid == self.osmid


class OrderLocation(GeoLocation):
    PICK_UP_TYPE = 0
    DROP_OFF_TYPE = 1

    __slots__ = ["order_location_type", "belong_to_order"]

    def __init__(self, osmid: int, x: int, y: int, location_type: int):
        super().__init__(osmid, x, y)
        self.order_location_type = location_type

    def set_order_belong(self, order):
        setattr(self, "belong_to_order", order)

    def __hash__(self):
        return hash((self.belong_to_order.order_id, self.order_location_type))

    def __repr__(self):
        if self.order_location_type == OrderLocation.PICK_UP_TYPE:
            location_type = "PICK"
        else:
            location_type = "DROP"

        return "({0},{1})".format(super().__repr__(), location_type)


class Order:
    __slots__ = ["order_id", "start_location", "end_location", "request_time", "max_wait_time", "order_distance",
                 "trip_fare", "detour_ratio", "n_riders", "pick_up_distance", "belong_to_vehicle"]

    def __init__(self, order_id: int, start_location: OrderLocation, end_location: OrderLocation,
                 request_time: int, max_wait_time: int, order_distance, trip_fare: float,
                 detour_ratio: float, n_riders: int):
        """
        构造函数
        :param order_id: 订单编号
        :param start_location: 起始地
        :param end_location: 终止地
        :param request_time: 请求时间
        :param max_wait_time: 最大等待时间
        :param order_distance: 订单行程距离
        :param trip_fare: 订单费用
        :param detour_ratio: 绕路比率
        :param n_riders: 订单的乘客
        """
        self.order_id = order_id
        self.start_location = start_location
        self.start_location.set_order_belong(self)
        self.end_location = end_location
        self.end_location.set_order_belong(self)
        self.request_time = request_time
        self.max_wait_time = max_wait_time
        self.order_distance = order_distance
        self.trip_fare = trip_fare
        self.detour_ratio = detour_ratio
        self.n_riders = n_riders
        self.pick_up_distance = 0.0  # 归属车辆接乘客已经行驶的里程
        self.belong_to_vehicle = None  # 订单归属车辆

    def __eq__(self, other):
        return other.order_id == self.order_id

    def __hash__(self):
        return hash(self.order_id)

    def __repr__(self):
        # return "{0} -> {1} fare:{2} detour_ratio:{3} rider_number:{4}". \
        #     format(self.start_location, self.end_location, self.trip_fare, self.detour_ratio, self.n_riders)
        return "start_time in {0} max_wait_time {1}, rider_number: {2}".format(self.request_time, self.max_wait_time,
                                                                               self.n_riders)


class Vehicle:
    WITHOUT_MISSION_STATUS = 0
    HAVE_MISSION_STATUS = 1
    AVERAGE_SPEED = 1.609344 * 12 / 60

    __slots__ = ["vehicle_id", "location", "n_seats", "route_plan", "cost_per_km", "status", "trip_distance"]

    def __init__(self, vehicle_id: int, location: GeoLocation, n_seats: int, cost_per_km: float, status: int):
        """
        构造函数
        :param vehicle_id: 车辆id
        :param location: 车辆当前位置
        :param n_seats:  车辆剩下的位置数目
        :param cost_per_km: 车俩每一公里的成本
        :param status: 车辆的状态
        """
        self.vehicle_id = vehicle_id
        self.location = location
        self.n_seats = n_seats
        self.cost_per_km = cost_per_km
        self.status = status
        self.route_plan = []
        self.trip_distance = 0.0

    @classmethod
    def set_average_speed(cls, average_speed: float):
        cls.AVERAGE_SPEED = average_speed

    def compute_cost(self, route_plan: List[OrderLocation], current_time: int) \
            -> float:
        """
        计算车辆的当前的位置按照某一个路线的成本
        :param route_plan: 订单坐标执行顺序
        :param current_time: 当前时间
        :return:
        """
        # 这里面所有操作都不可以改变自身原有的变量的值 ！！！！！
        trip_distance = self.trip_distance
        current_trip_time = current_time
        current_trip_cost = 0.0

        vehicle_location = [self.location.x, self.location.y]
        for order_location in route_plan:
            two_location_distance = np.abs(vehicle_location[0] - order_location.x) \
                                    + np.abs(vehicle_location[1] - order_location.y)
            trip_distance += two_location_distance  # 行驶历程
            current_trip_time += (two_location_distance / Vehicle.AVERAGE_SPEED)  # minute
            current_trip_cost += self.cost_per_km * two_location_distance  # 计算成本

            belong_to_order = order_location.belong_to_order  # 订单坐标隶属的订单
            if order_location.order_location_type == OrderLocation.PICK_UP_TYPE:
                if current_trip_time > belong_to_order.request_time + belong_to_order.max_wait_time:
                    return np.inf  # 无法满足max_wait_minute返回无穷大

                belong_to_order.pick_up_distance = trip_distance  # 更新接乘客已经行驶的里程

            if order_location.order_location_type == OrderLocation.DROP_OFF_TYPE:
                real_order_distance = trip_distance - belong_to_order.pick_up_distance  # 计算绕路比
                if real_order_distance > belong_to_order.order_distance * (1 + belong_to_order.detour_ratio):
                    return np.inf  # 无法满足绕路比返回无穷大

            # 更新当前车辆坐标
            vehicle_location[0] = order_location.x
            vehicle_location[1] = order_location.y

        return current_trip_cost

    def find_route_plan(self, order: Order, current_time: int, strategy="min") \
            -> Tuple[float, List[OrderLocation]]:
        """
        利用保持原有路线顺序而插入新的的订单的起始地和结束地，返回成本最优的插入情况
        返回插入之后的成本和之后的插入情况
        :param order: 订单
        :param current_time: 当前时间
        :param strategy: 策略
        :return:
        """
        # 这里面所有操作都不可以改变自身原有的变量的值 ！！！！！
        if len(self.route_plan) == 0:  # 如果以前没有乘客在车上
            best_route_plan = [order.start_location, order.end_location]
            best_cost = self.compute_cost(best_route_plan, current_time)
        else:  # 如果不是空的的就枚举插入
            start_location = order.start_location
            end_location = order.end_location
            route_plan = self.route_plan
            if strategy == "min":
                best_cost, best_route_plan = np.inf, []
                for i in range(len(self.route_plan) + 1):
                    for j in range(i, len(self.route_plan) + 1):
                        temp_route_plan = route_plan[:i] + [start_location] + \
                                          route_plan[i:j] + [end_location] + \
                                          route_plan[j:]
                        temp_cost = self.compute_cost(temp_route_plan, current_time)
                        if temp_cost < best_cost:
                            best_cost = temp_cost
                            best_route_plan = temp_route_plan
            else:
                best_cost, best_route_plan = -np.inf, []
                for i in range(len(self.route_plan) + 1):
                    for j in range(len(self.route_plan) + 1):
                        temp_route_plan = route_plan[:i] + [start_location] + \
                                          route_plan[i:j] + [end_location] + \
                                          route_plan[j:]
                        temp_cost = self.compute_cost(temp_route_plan, current_time)
                        if temp_cost != np.inf:
                            if temp_cost > best_cost:
                                best_cost = temp_cost
                                best_route_plan = temp_route_plan

        return best_cost, best_route_plan

    def update_random_location(self, row, col):
        """
        随机更新车辆的下一分钟会到达的节点
        :return:
        """
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        inds = [0, 1, 2, 3]
        distance = 0
        while True:
            i = np.random.choice(inds)
            if self.location.x + dirs[i][0] >= row or self.location.x + dirs[i][0] < 0 or \
                    self.location.y + dirs[i][1] >= col or self.location.y + dirs[i][1] < 0:
                continue
            self.location.x += dirs[i][0]
            self.location.y += dirs[i][1]
            distance += 1
            if distance == Vehicle.AVERAGE_SPEED:
                self.trip_distance += distance
                break

    def update_order_location(self):
        """
        计算车辆按照规划路线的下一步会到达的节点
        :return:
        """
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def __get_distance__(l1, l2):
            return np.abs(l1[0] - l2[0]) + np.abs(l1[1] - l2[1])

        def __get_next_x_y__(location1, location2):
            pre_distance = __get_distance__((location1.x, location1.y), (location2.x, location2.y))
            for i in range(4):
                x = location1.x + dirs[i][0]
                y = location1.y + dirs[i][1]
                cur_distance = __get_distance__((x, y), (location2.x, location2.y))
                if cur_distance < pre_distance:
                    return x, y

        un_covered_location_index = 0  # 未完成订单坐标的最小索引
        average_speed = Vehicle.AVERAGE_SPEED
        current_trip_distance = 0.0  # 模拟当前这一分钟的路程情况而进行的设定
        for index, order_location in enumerate(self.route_plan):
            two_location_distance = __get_distance__((self.location.x, self.location.y),
                                                     (order_location.x, order_location.y))
            if current_trip_distance + two_location_distance - average_speed <= 0.01:  # 首先判断是否可以一分钟是否过剩
                current_trip_distance += two_location_distance  # 更新当前行程的路径
                self.location.x = order_location.x
                self.location.y = order_location.y  # 更新车辆坐标
                un_covered_location_index = index + 1  # 未完成订单坐标索引增加
                belong_to_order = order_location.belong_to_order
                if order_location.order_location_type == OrderLocation.PICK_UP_TYPE:  # 更新接乘客的形成路径
                    belong_to_order.pick_up_distance = self.trip_distance + current_trip_distance
                if order_location.order_location_type == OrderLocation.DROP_OFF_TYPE:  # 更新车辆可用座位数量
                    self.n_seats += belong_to_order.n_riders
                if current_trip_distance + two_location_distance - average_speed >= -0.01:  # 行程刚好满足一分钟的情况
                    break
            else:  # 订单节点路长过大无法到达的情况
                un_covered_location_index = index
                while True:
                    x, y = __get_next_x_y__(self.location, order_location)
                    partial_two_location_distance = __get_distance__((self.location.x, self.location.y), (x, y))
                    if current_trip_distance + partial_two_location_distance - average_speed <= 0.01:
                        current_trip_distance += partial_two_location_distance
                        self.location.x = x
                        self.location.y = y
                    else:  # 选择最接近1min的路程
                        if 2 * current_trip_distance + partial_two_location_distance < 2 * average_speed:
                            self.location.x = x
                            self.location.y = y
                        break
                break

        self.route_plan = self.route_plan[un_covered_location_index:]
        self.trip_distance += current_trip_distance
        if len(self.route_plan) != 0:
            self.status = Vehicle.HAVE_MISSION_STATUS
        else:
            self.status = Vehicle.WITHOUT_MISSION_STATUS

    def __repr__(self):
        return "id is {0} location = {1} available_seats = {2} trip_distance = {3} route_plan = {4}". \
            format(self.vehicle_id, self.location, self.n_seats, self.trip_distance, self.route_plan)

    def __hash__(self):
        return hash(self.vehicle_id)

    def __eq__(self, other):
        return self.vehicle_id == other.vehicle_id


if __name__ == '__main__':
    # ol1 = OrderLocation(0, 1, 1, OrderLocation.PICK_UP_TYPE)
    # ol2 = OrderLocation(1, 1, 2, OrderLocation.DROP_OFF_TYPE)
    # ol3 = OrderLocation(2, 2, 2, OrderLocation.PICK_UP_TYPE)
    # ol4 = OrderLocation(3, 3, 4, OrderLocation.DROP_OFF_TYPE)
    # o1 = Order(0, ol1, ol2, 0, 2, 1, 1, 0.5, 1)
    # o2 = Order(0, ol3, ol4, 0, 2, 3, 3, 0.5, 1)
    # route_plan = [ol3, ol4]
    # vehicle = Vehicle(0, GeoLocation(0, 1, 1), 2, 0.5, Vehicle.WITHOUT_MISSION_STATUS)
    # Vehicle.set_average_speed(4)
    # vehicle.route_plan = route_plan
    # vehicle.n_seats -= 1
    # c, r = vehicle.find_route_plan(o1, 0)
    # print(c, r)
    pass
