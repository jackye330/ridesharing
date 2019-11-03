#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/22
from typing import List

import numpy as np

from agent.order import Order
from network.transport_network.graph import NetworkGraph
from network.transport_network.location import VehicleLocation, PickLocation, DropLocation, GeoLocation
from setting import AVERAGE_SPEED
from setting import DISTANCE_EPS
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

    def drive_on_random(self, network: NetworkGraph):  # TODO 需要修改
        """
        车辆随机在路上行驶
        :param network: 交通路网
        :return:
        ------
        注意：
        不要那些只可以进去，不可以出来的节点
        如果车辆就正好在一个节点之上，那么随机选择一个节点到达，如果不是这些情况就在原地保持不动
        """
        should_drive_distance = AVERAGE_SPEED * TIME_SLOT  # 需要行驶的距离

        if self.location.driven_distance < 0:
            self.location.reset()

        if self.location.is_between:  # 车辆处于两个节点之间
            vehicle_to_goal_distance = network.shortest_distance[self.location.osm_index, self.location.goal_index]
            vehicle_to_goal_distance -= self.location.driven_distance

            if vehicle_to_goal_distance - should_drive_distance <= DISTANCE_EPS:  # 可以到goal_index上
                self.drive_distance += vehicle_to_goal_distance
                self.location.osm_index = self.location.goal_index
                self.location.reset()
                should_drive_distance -= vehicle_to_goal_distance
            else:
                self.drive_distance += should_drive_distance
                should_drive_distance = 0.0

            if should_drive_distance <= DISTANCE_EPS:
                return
            else:  # TODO 不好处理
                pass

        else:
            # 在规定的时间到不了任何节点到不了任何节点 (有的节点包含自己，有的不包含1265，3103)
            next_index_set = set(network.shortest_path_time_slot[self.location.osm_index, :])
            adjacent_index_set = set(network.shortest_path[self.location.osm_index, :])

            if len(next_index_set) == 0:
                pass
            else:
                next_location_index = np.random.choice(next_index_set)  # 随机选择下一个坐标
                if network.shortest_path_goal_index[self.location.osm_index, next_location_index]:
                    pass
                self.drive_distance += network.shortest_distance[self.location.osm_index, next_location_index]  # 更新行程数据
                self.location.osm_index = next_location_index  # 更新车辆坐标


            if len(next_index_set) <= 2 or (self.location.osm_index in next_index_set and len(next_index_set) <= 3):
                # 一分钟到不了任何节点，用相邻节点的方法，记录中间位置
                adjacent_nodes_list = list(set(network.shortest_path[self.location.osm_index, :]))
                # 没有相邻节点，车的位置随机选取
                if len(adjacent_nodes_list) <= 1:
                    self.location.reset()
                    self.location.osm_index = np.random.choice(len(adjacent_nodes))
                else:
                    next_location_index = np.random.choice(adjacent_nodes_list)
                    while next_location_index == -2:
                        next_location_index = np.random.choice(adjacent_nodes_list)
                    self.location.is_between = True
                    self.location.driven_distance += AVERAGE_SPEED
                    self.location.goal_index = next_location_index
            else:
                pass

        # TODO 这个之后需要被革除掉
        next_index_set = list(set(next_location_index_with_time_slot[self.location.osm_index, :]))
        if self.location.is_between:  # 处于两个点中间
            two_location_distance = network.shortest_distance[self.location.osm_index, self.location.goal_index]

            if self.location.driven_distance - two_location_distance < -DISTANCE_EPS:
                self.location.driven_distance += should_drive_distance
            else:
                while self.location.driven_distance >= 20.0:
                    two_location_distance = network.shortest_distance[self.location.osm_index, self.location.goal_index]
                    if self.location.driven_distance - two_location_distance >= -DISTANCE_EPS:
                        self.drive_distance += two_location_distance
                        self.location.osm_index = self.location.goal_index
                        self.location.driven_distance -= two_location_distance
                        adjacent_nodes_list = list(set(adjacent_nodes[self.location.osm_index, :]))
                        # 没有相邻节点，随机跳到一个节点
                        if len(adjacent_nodes_list) <= 1:
                            self.location.reset()
                            self.location.osm_index = np.random.choice(len(adjacent_nodes))
                            break
                        next_location_index = np.random.choice(adjacent_nodes_list)
                        while next_location_index == -1 or next_location_index == -2:
                            next_location_index = np.random.choice(next_index_set)
                        self.location.goal_index = next_location_index
                    else:
                        break
                if self.location.driven_distance < 20.0:
                    self.location.reset()
                    self.location.driven_distance = 0.0
        else:
            pass

    def drive_on_route_plan(self, network: NetworkGraph):
        """
        车辆自己按照自己的路径规划行驶
        :param network: 交通路网
        """
        un_covered_location_index = 0  # 未完成订单坐标的最小索引
        should_drive_distance = AVERAGE_SPEED * TIME_SLOT  # 当前时间间隙内需要行驶的距离

        g = network.real_vehicle_drive_on_route_plan(self.location, self.route_plan, should_drive_distance)
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
                    self.n_seats += belong_to_order.n_riders  # 更新车辆可用座位数量

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


if __name__ == '__main__':
    # TODO: 测试agent的功能
    l_m = {0: (0.0, 0.0), 1: (0.0, 1.0), 2: (0.0, 2.0), 3: (0.0, 3.0),
           4: (1.0, 0.0), 5: (1.0, 1.0), 6: (1.0, 2.0), 7: (1.0, 3.0),
           8: (2.0, 0.0), 9: (2.0, 1.0), 10: (2.0, 2.0), 11: (2.0, 3.0),
           12: (3.0, 0.0), 13: (3.0, 1.0), 14: (3.0, 2.0), 15: (3.0, 3.0)}
    import math
    import numpy as np

    s_d = np.zeros(shape=(len(l_m), len(l_m)), dtype=np.float32)
    for i in range(len(l_m)):
        for j in range(len(l_m)):
            s_d[i, j] = math.sqrt((l_m[i][0] - l_m[j][0]) ** 2 + (l_m[i][1] - l_m[j][1]) ** 2)

    s_p = np.zeros(shape=(len(l_m), len(l_m)), dtype=np.float32)
    for i in range(len(l_m)):
        for j in range(len(l_m)):
            s_p[i, j] = j

    GeoLocation.set_location_map(l_m)
    Vehicle.set_average_speed(1.0)
    t = 0
    p0 = VehicleLocation(0)
    p1 = PickLocation(5)
    p2 = DropLocation(6)
    p3 = PickLocation(8)
    p4 = DropLocation(14)
    p5 = PickLocation(11)
    p6 = DropLocation(15)

    # order_id, start_location, end_location, request_time, max_wait_time,
    # order_distance, trip_fare, detour_ratio, n_riders
    distance_1 = s_d[p1.osm_index, p2.osm_index]
    order1 = Order(1, p1, p2, t, 10, distance_1, 3.2 * distance_1, 1.0, 1)
    distance_2 = s_d[p3.osm_index, p4.osm_index]
    order2 = Order(2, p3, p4, t, 5, distance_2, 2.1 * distance_2, 1.0, 1)
    distance_3 = s_d[p5.osm_index][p6.osm_index]
    order3 = Order(3, p5, p6, t, 50, distance_3, 4 * distance_3, 1.0, 1)
    print(order1)
    print(order2)
    distance_o1_o2 = s_d[order1.drop_location.osm_index][order2.pick_location.osm_index]
    print("o2", distance_2)
    print("o1_o2", distance_o1_o2)

    # # vehicle_id: int, location: GeoLocation, n_seats: int, cost_per_km: float, status
    vehicle1 = Vehicle(0, p0, 4, 1.0)

    rem_list = []
    rem_list.append(order1.pick_location)
    rem_list.append(order2.pick_location)

    profit, schedule = vehicle1.find_best_schedule([], rem_list, s_d, 0, 0.0, [])
    print(vehicle1)
    # vehicle1.route_plan = schedule
    # rem_list = vehicle1.get_rem_list(schedule.copy())
    # rem_list.append(order3.start_location)
    # print("add a new order")
    # profit, schedule = vehicle1.find_best_schedule([], rem_list.copy(), s_d, 0, 0.0, [])
    print("best_profit", profit)
    print("best_schedule", schedule)

    # vehicle1.status = Vehicle.HAVE_MISSION_STATUS
    # vehicle1.n_seats -= 1
    # c, r = vehicle1.find_route_plan(s_d, order1, t)
    # print(vehicle1.location)
    # print(c)
    # print(r)
    # vehicle1.route_plan = r
    # c, r = vehicle1.find_route_plan(s_d, order2, t)
    # print(c)
    # print(r)
    # print(vehicle1.location)
    # vehicle1.route_plan = r
    #
    # # update算法是否正常
    # Vehicle.set_average_speed(5.0)
    # print(vehicle1)
    # vehicle1.drive_on_route_plan(s_d, s_p)
    # print(vehicle1)
