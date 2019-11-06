#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/22
from inspect import getgeneratorstate
from env.location import VehicleLocation
from env.location import PickLocation
from env.location import DropLocation
from env.network import Network
from setting import TIME_SLOT
from constant import FLOAT_ZERO
from constant import FIRST_INDEX

__all__ = ["Vehicle"]


class Vehicle:
    __slots__ = ["vehicle_id", "location", "available_seats", "unit_cost", "route_plan", "drive_distance", "is_activated"]
    average_speed = None  # 车辆平均速度
    bidding_strategy = None  # 车辆投标策略

    def __init__(self, vehicle_id: int, location: VehicleLocation, available_seats: int, unit_cost: float):
        """
        构造函数
        :param vehicle_id: 车辆id
        :param location: 车辆当前位置
        :param available_seats:  车辆剩下的位置数目
        :param unit_cost: 车俩的单位行驶成本
        """
        self.vehicle_id = vehicle_id  # 车辆唯一标识
        self.location = location      # 车辆当前的位置
        self.available_seats = available_seats        # 剩下的座位数目
        self.unit_cost = unit_cost    # 单位行驶成本
        self.route_plan = []          # 车辆自身的路径规划，默认没有订单位置在路径规划上
        self.is_activated = True      # 车辆是否处于激活状态，默认车是激活状态
        self.drive_distance = FLOAT_ZERO  # 车辆总行驶距离, 默认没有任何行驶记录

    @classmethod
    def set_average_speed(cls, average_speed):
        cls.average_speed = average_speed

    @classmethod
    def set_bidding_strategy(cls, strategy):
        cls.bidding_strategy = strategy

    @classmethod
    def generate_vehicles(cls, network):
        """
        用于生成一组自己的集合
        :param network:
        :return:
        """
        from setting import VEHICLE_SPEED
        from setting import VEHICLE_NUMBER
        from setting import VEHICLE_FUEL_COST_RATIO
        from setting import FUEL_CONSUMPTION_DATA_FILE
        import pandas as pd
        Vehicle.set_average_speed(VEHICLE_SPEED)  # 初初始化车辆速度
        locations = network.get_random_vehicle_locations(VEHICLE_NUMBER)  # 得到车辆位置
        car_fuel_consumption_info = pd.read_csv(FUEL_CONSUMPTION_DATA_FILE)
        cars_info = car_fuel_consumption_info.sample(n=VEHICLE_NUMBER)
        vehicles = []
        for vehicle_id in range(VEHICLE_NUMBER):
            location = locations[vehicle_id]
            car_info = cars_info.iloc[vehicle_id, :]
            n_seats = int(car_info["seats"])
            unit_cost = float(car_info["fuel_consumption"]) * VEHICLE_FUEL_COST_RATIO
            vehicle = cls(vehicle_id, location, n_seats, unit_cost)
            vehicles.append(vehicle)
        return vehicles

    def drive_on_random(self, network: Network):
        """
        车辆随机在路上行驶
        :param network: 交通路网
        :return:
        ------
        注意：
        不要那些只可以进去，不可以出来的节点
        如果车辆就正好在一个节点之上，那么随机选择一个节点到达，如果不是这些情况就在原地保持不动
        """
        could_drive_distance = self.average_speed * TIME_SLOT
        self.drive_distance += network.drive_on_random(self.location, could_drive_distance)

    def drive_on_route_plan(self, network: Network):
        """
        车辆自己按照自己的路径规划行驶
        :param network: 交通路网
        """
        un_covered_location_index = FIRST_INDEX  # 未完成订单坐标的最小索引
        could_drive_distance = self.average_speed * TIME_SLOT
        g = network.drive_on_route_plan(self.location, self.route_plan, could_drive_distance)  # 开启生成器
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
                    # self.available_seats -= belong_to_order.n_riders  # 这种更新适用与动态更新车辆可用座位数量
                if isinstance(order_location, DropLocation):
                    self.available_seats += belong_to_order.n_riders  # 因为送到乘客了，可以腾出位置

        self.route_plan = self.route_plan[un_covered_location_index:]  # 更新自己的路线规划

        if getgeneratorstate(g) != "GEN_CLOSED":  # 如果生成器没有关闭要自动关闭
            g.close()

    def __repr__(self):
        return "id is {0} location is {1} available_seats is {2} drive_distance is {3} unit_cost is {4} route_plan = {5}". \
            format(self.vehicle_id, self.location, self.available_seats, self.drive_distance, self.unit_cost, self.route_plan)

    def __hash__(self):
        return hash(self.vehicle_id)

    def __eq__(self, other):
        return self.vehicle_id == other.vehicle_id
