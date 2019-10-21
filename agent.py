#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/3/3
from typing import List, Tuple, Dict

import numpy as np


class GeoLocation:
    LOCATION_MAP = {}  # 自己坐标的运动体系index->osm_id->(longitude, latitude)
    INDEX2OSM_ID = {}  # 自己坐标系的索引转换index->osm_id

    __slots__ = ["osm_index"]

    def __init__(self, osm_index: int):
        """
        :param osm_index: open street map id 字典的索引
        """
        self.osm_index = osm_index

    @classmethod
    def set_location_map(cls, location_map: Dict[int, Tuple[float, float]]):
        cls.LOCATION_MAP = location_map

    @classmethod
    def set_index2osm_id(cls, index2osm_id: Dict[int, int]):
        cls.INDEX2OSM_ID = index2osm_id

    def __repr__(self):
        # longitude = GeoLocation.LOCATION_MAP[self.osm_index][0]
        # latitude = GeoLocation.LOCATION_MAP[self.osm_index][1]
        # return "({0:6},{1:6})".format(longitude, latitude)
        return "{0}".format(self.osm_index)

    def __hash__(self):
        return hash(self.osm_index)

    def __eq__(self, other):
        return other.osm_index == self.osm_index


class OrderLocation(GeoLocation):
    PICK_UP_TYPE = 0
    DROP_OFF_TYPE = 1

    __slots__ = ["order_location_type", "belong_to_order"]

    def __init__(self, osm_index: int, location_type: int):
        """
        :param osm_index:
        :param location_type:
        """
        super().__init__(osm_index)
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
        return "start_time in {0} max_wait_time {1}, rider_number: {2}".format(self.request_time, self.max_wait_time, self.n_riders)


class Vehicle:
    WITHOUT_MISSION_STATUS = 0
    HAVE_MISSION_STATUS = 1
    POSITIVE = 1
    NEGATIVE = 0
    IS_BETWEEN_TWO_INDEX = 1
    NOT_BETWEEN_TWO_INDEX = 0
    AVERAGE_SPEED = 1.609344 * 12 / 60

    __slots__ = ["vehicle_id", "location", "n_seats", "route_plan", "cost_per_km", "status", "trip_distance", "activated", "is_between", "between_distance", "goal_index"]

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
        self.activated = Vehicle.POSITIVE   # 默认车都是激活状态
        self.is_between = Vehicle.NOT_BETWEEN_TWO_INDEX     # 默认不在两个没有间隔的index中间
        self.between_distance = 0.0     # 距离上一个index节点的距离
        self.goal_index = -1        # 在两个点中间时的目标点osm_index
        self.trip_distance = 0.0

    @classmethod
    def set_average_speed(cls, average_speed: float):
        cls.AVERAGE_SPEED = average_speed

    def compute_cost(self, short_distance: np.ndarray, route_plan: List[OrderLocation], current_time: int) \
            -> float:
        """
        计算车辆的当前的位置按照某一个路线的成本
        :param short_distance: 最短路径矩阵
        :param route_plan: 订单坐标执行顺序
        :param current_time: 当前时间
        :return:
        """
        # 这里面所有操作都不可以改变自身原有的变量的值 ！！！！！
        trip_distance = self.trip_distance
        current_trip_time = current_time
        current_trip_cost = 0.0
        vehicle_location_osm_index = self.location.osm_index
        if len(route_plan) == 0:
            return current_trip_cost
        # 如果车在两个没有中间节点的节点中间
        if self.is_between == Vehicle.IS_BETWEEN_TWO_INDEX:
            # 顺路或者不能反向行驶
            if short_distance[self.goal_index, route_plan[0].osm_index] + \
                    short_distance[vehicle_location_osm_index, self.goal_index] - self.between_distance < \
                    short_distance[vehicle_location_osm_index, route_plan[0].osm_index] + self.between_distance \
                    or short_distance[self.goal_index, vehicle_location_osm_index] == np.inf:
                vehicle_location_osm_index = self.goal_index
                current_trip_time += (short_distance[vehicle_location_osm_index, self.goal_index] -
                                      self.between_distance) / Vehicle.AVERAGE_SPEED
                current_trip_cost += self.cost_per_km * (short_distance[vehicle_location_osm_index, self.goal_index] -
                                                         self.between_distance)
                trip_distance += short_distance[vehicle_location_osm_index, self.goal_index] - self.between_distance
            else:
                current_trip_time += (self.between_distance / Vehicle.AVERAGE_SPEED)
                current_trip_cost += self.cost_per_km * self.between_distance
                trip_distance += self.between_distance
        for order_location in route_plan:
            two_location_distance = short_distance[vehicle_location_osm_index, order_location.osm_index]
            trip_distance += two_location_distance  # 行驶历程
            current_trip_time += (two_location_distance / Vehicle.AVERAGE_SPEED)  # minute
            current_trip_cost += self.cost_per_km * two_location_distance  # 计算成本

            belong_to_order = order_location.belong_to_order  # 订单坐标隶属的订单（这样赋值是浅复制？）
            if order_location.order_location_type == OrderLocation.PICK_UP_TYPE:
                if current_trip_time > belong_to_order.request_time + belong_to_order.max_wait_time:
                    return np.inf  # 无法满足max_wait_minute返回无穷大

                belong_to_order.pick_up_distance = trip_distance  # 更新接乘客已经行驶的里程

            if order_location.order_location_type == OrderLocation.DROP_OFF_TYPE:
                real_order_distance = trip_distance - belong_to_order.pick_up_distance  # 计算绕路比
                if real_order_distance > belong_to_order.order_distance * (1 + belong_to_order.detour_ratio):
                    return np.inf  # 无法满足绕路比返回无穷大

            vehicle_location_osm_index = order_location.osm_index  # 更新当前车辆坐标

        return current_trip_cost

    def find_route_plan(self, short_distance: np.ndarray, order: Order, current_time: int, strategy="min") \
            -> Tuple[float, List[OrderLocation]]:
        """
        利用保持原有路线顺序而插入新的的订单的起始地和结束地，返回成本最优的插入情况
        返回插入之后的成本和之后的插入情况
        :param short_distance: 最短路径
        :param order: 订单
        :param current_time: 当前时间
        :param strategy: 策略
        :return:
        """
        # 这里面所有操作都不可以改变自身原有的变量的值 ！！！！！
        if len(self.route_plan) == 0:  # 如果以前没有乘客在车上
            best_route_plan = [order.start_location, order.end_location]
            best_cost = self.compute_cost(short_distance, best_route_plan, current_time)
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
                        temp_cost = self.compute_cost(short_distance, temp_route_plan, current_time)
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
                        temp_cost = self.compute_cost(short_distance, temp_route_plan, current_time)
                        if temp_cost != np.inf:
                            if temp_cost > best_cost:
                                best_cost = temp_cost
                                best_route_plan = temp_route_plan

        return best_cost, best_route_plan

    def find_best_schedule(self, cur_list: List[OrderLocation], rem_list: List[OrderLocation], short_distance: np.ndarray,
                           current_time: int, best_profit: float, best_cost: float, best_schedule: List[OrderLocation]) \
            -> Tuple[float, float, List[OrderLocation]]:
        """
                Mohammad Asghari 论文里面的找路径规划的方法，可以打乱原来的先后顺序
        :param cur_list:
        :param rem_list:
        :param short_distance:
        :param current_time:
        :param best_profit:
        :param best_cost:
        :param best_schedule:
        :return: best_profit:
        :return: best_cost:
        :return: best_schedule:
        """
        #   所有操作不能改变self的值
        # print("cur_list", cur_list)
        # print("rem_list", rem_list)
        for p in rem_list:
            cur_list_ = cur_list.copy()
            cur_list_.append(p)
            # print("new_cur_list", cur_list_)
            profit, cost = self.get_profit(cur_list_, short_distance, current_time)
            if profit != - np.inf:
                rem_list_ = rem_list.copy()
                rem_list_.remove(p)
                if p.order_location_type == OrderLocation.PICK_UP_TYPE:
                    rem_list_.append(p.belong_to_order.end_location)
                if len(rem_list_) == 0:
                    return profit, cost, cur_list_
                profit, cost, schedule = self.find_best_schedule(cur_list_, rem_list_, short_distance,
                                                           current_time, best_profit, best_cost, best_schedule.copy())
                if profit > best_profit:
                    best_profit = profit
                    best_cost = cost
                    best_schedule = schedule
            cur_list_.remove(p)
        return best_profit, best_cost, best_schedule

    def get_rem_list(self, schedule: List[OrderLocation]) \
            -> List[OrderLocation]:
        """
            构造rem_list 列表 ：只包含起始点或者单独的终点
        :param schedule:
        :return:
        """
        if len(schedule) == 0:
            return schedule
        rem_list = []
        rem_order_list = []
        for p in schedule:
            # p 是 起始点 直接加入
            if p.order_location_type == OrderLocation.PICK_UP_TYPE:
                rem_list.append(p)
                rem_order_list.append(p.belong_to_order)
            else:
                # p 是终点 先判断rem_list里面有没有对应的起始点
                if rem_order_list.__contains__(p.belong_to_order):
                    continue
                else:   # 没有就加入
                    rem_list.append(p)
        return rem_list

    # 需要返回利润和成本
    def get_profit(self, schedule: List[OrderLocation], short_distance: np.ndarray, current_time: int) \
            -> Tuple[float, float]:
        """
            计算schedule 路径规划下的对平台的收益
        :param schedule:
        :param short_distance:
        :param current_time:
        :return: profit:
        :return: cost:
        """
        #   所有操作不能改变self的值 所有过程都是模拟的，车实际上还没有运行到这些点
        # print("schedule", schedule)
        if len(schedule) == 0:
            return 0.0, 0.0
        time = current_time
        distance = 0.0
        fare = 0.0
        trip_distance = self.trip_distance
        current_location_index = self.location.osm_index
        if self.is_between == Vehicle.IS_BETWEEN_TWO_INDEX:
            # 顺路或者不能反向行驶
            if short_distance[self.goal_index, schedule[0].osm_index] + \
                    short_distance[current_location_index, self.goal_index] - self.between_distance < \
                    short_distance[current_location_index, schedule[0].osm_index] + self.between_distance \
                    or short_distance[self.goal_index, current_location_index] == np.inf:
                current_location_index = self.goal_index
                time += (short_distance[current_location_index, self.goal_index] -
                         self.between_distance) / Vehicle.AVERAGE_SPEED
                trip_distance += short_distance[current_location_index, self.goal_index] - self.between_distance
            else:
                time += (self.between_distance / Vehicle.AVERAGE_SPEED)
                trip_distance += self.between_distance
        pick_up = {}
        for p in schedule:
            trip = short_distance[current_location_index][p.osm_index]
            time += trip / self.AVERAGE_SPEED
            distance += trip
            trip_distance += trip   # 记录车从开始到现在走过的路程 用来计算订单实际走过的路程
            # 判断能否按时接到
            if p.order_location_type == OrderLocation.PICK_UP_TYPE:
                if time > p.belong_to_order.request_time + p.belong_to_order.max_wait_time:
                    # print("over time")
                    return -np.inf, -np.inf
                pick_up[p.belong_to_order] = distance   # 能接到，记录当前的位置
            if p.order_location_type == OrderLocation.DROP_OFF_TYPE:    # 这一分钟没有pick_up点 pick_up[p.belong_to_order]没有记录
                # 判断路径规划s 和 e 位置是否合理
                if not pick_up.__contains__(p.belong_to_order):
                    # print("unavailable")
                    # 上几阶段接到的订单 所以pick_up{}里面没有相应的数据
                    if p.belong_to_order.pick_up_distance < trip_distance:
                        delta_d = trip_distance - p.belong_to_order.pick_up_distance - p.belong_to_order.order_distance
                    else:
                        return -np.inf, -np.inf
                else:
                    delta_d = distance - pick_up[p.belong_to_order] - p.belong_to_order.order_distance     # 计算订单绕路路程
                # 判断绕路比是否满足
                if delta_d > p.belong_to_order.detour_ratio * p.belong_to_order.order_distance:
                    # print("over detour", p.belong_to_order.order_id)
                    return -np.inf, -np.inf
                # 折扣按照 f = 1 - (0.25 * delta_d^2) 计算
                fare += p.belong_to_order.trip_fare * (1 - (0.00025 * np.power(delta_d, 2)))
                # print("fare", fare)
            current_location_index = p.osm_index
        cost = self.cost_per_km * distance
        # print("cost", cost)
        profit = fare - cost
        # print("profit", profit)
        return profit, cost

    def update_random_location(self, shortest_distance: np.ndarray, shortest_path_with_minute: np.ndarray,
                               adjacent_nodes: np.ndarray):
        """
        随机更新车辆的下一分钟会到达的节点
        :param shortest_distance: 最短路径长度矩阵
        :param shortest_path_with_minute: 最短路径下一分钟位置矩阵
        :param adjacent_nodes: 每个节点的相邻节点集合
        :return:
        """
        next_index_set = list(set(shortest_path_with_minute[self.location.osm_index, :]))
        # print(self.location.osm_index, next_index_set)
        if self.between_distance < 0:
            self.is_between = Vehicle.NOT_BETWEEN_TWO_INDEX
            self.between_distance = 0
            self.goal_index = -1
        if self.is_between == Vehicle.IS_BETWEEN_TWO_INDEX:     # 处于两个点中间
            two_location_distance = shortest_distance[self.location.osm_index][self.goal_index]
            if self.between_distance - two_location_distance < -0.01:
                self.between_distance += Vehicle.AVERAGE_SPEED
            else:
                while self.between_distance >= 0.2:
                    two_location_distance = shortest_distance[self.location.osm_index][self.goal_index]
                    if self.between_distance - two_location_distance >= -0.01:
                        self.trip_distance += two_location_distance
                        self.location.osm_index = self.goal_index
                        self.between_distance -= two_location_distance
                        adjacent_nodes_list = list(set(adjacent_nodes[self.location.osm_index]))
                        # 没有相邻节点，随机跳到一个节点
                        if len(adjacent_nodes_list) <= 1:
                            self.is_between = Vehicle.NOT_BETWEEN_TWO_INDEX
                            self.between_distance = 0
                            self.goal_index = -1
                            self.location.osm_index = np.random.choice(len(adjacent_nodes))
                            break
                        next_location_index = np.random.choice(adjacent_nodes_list)
                        while next_location_index == -1 or next_location_index == -2:
                            next_location_index = np.random.choice(next_index_set)
                        self.goal_index = next_location_index
                    else:
                        break
                if self.between_distance < 0.2:
                    self.is_between = Vehicle.NOT_BETWEEN_TWO_INDEX
                    self.between_distance = 0
                    self.goal_index = -1
        else:
            # 一分钟到不了任何节点 (有的节点包含自己，有的不包含1265，3103)
            if len(next_index_set) <= 2 or (self.location.osm_index in next_index_set and len(next_index_set) <= 3):
                # 一分钟到不了任何节点，用相邻节点的方法，记录中间位置
                adjacent_nodes_list = list(set(adjacent_nodes[self.location.osm_index]))
                # 没有相邻节点，车的位置随机选取
                if len(adjacent_nodes_list) <= 1:
                    self.is_between = Vehicle.NOT_BETWEEN_TWO_INDEX
                    self.between_distance = 0
                    self.goal_index = -1
                    self.location.osm_index = np.random.choice(len(adjacent_nodes))
                else:
                    next_location_index = np.random.choice(adjacent_nodes_list)
                    while next_location_index == -2:
                        next_location_index = np.random.choice(adjacent_nodes_list)
                    self.is_between = Vehicle.IS_BETWEEN_TWO_INDEX
                    self.between_distance += Vehicle.AVERAGE_SPEED
                    self.goal_index = next_location_index
            else:
                next_location_index = np.random.choice(next_index_set)  # 随机选择下一个坐标
                while next_location_index == -1 or next_location_index == -2:
                    next_location_index = np.random.choice(next_index_set)
                self.trip_distance += shortest_distance[self.location.osm_index, next_location_index]  # 更新行程数据
                self.location.osm_index = next_location_index  # 更新车辆坐标

    def update_order_location(self, shortest_distance: np.ndarray, shortest_path: np.ndarray):
        """
        计算车辆按照规划路线的下一步会到达的节点
        :param shortest_distance: 最短路径长度矩阵
        :param shortest_path: 最短路径矩阵
        :return:
        """
        un_covered_location_index = 0  # 未完成订单坐标的最小索引
        average_speed = Vehicle.AVERAGE_SPEED
        current_trip_distance = 0.0  # 模拟当前这一分钟的路程情况而进行的设定
        if self.between_distance < 0:
            self.is_between = Vehicle.NOT_BETWEEN_TWO_INDEX
            self.between_distance = 0
            self.goal_index = -1
        for index, order_location in enumerate(self.route_plan):
            two_location_distance = shortest_distance[self.location.osm_index, order_location.osm_index]
            # 如果在两个节点中间
            if self.is_between == Vehicle.IS_BETWEEN_TWO_INDEX:
                # 如果目标节点近或者为单行道不能返回
                if shortest_distance[self.goal_index, order_location.osm_index] + \
                        shortest_distance[self.location.osm_index, self.goal_index] - self.between_distance < \
                        shortest_distance[self.location.osm_index, order_location.osm_index] + self.between_distance \
                        or shortest_distance[self.goal_index, self.location.osm_index] == np.inf:
                    # 能到目标节点
                    if current_trip_distance + shortest_distance[self.location.osm_index, self.goal_index] - self.between_distance - average_speed <= 0.01:
                        current_trip_distance += shortest_distance[self.location.osm_index, self.goal_index] - self.between_distance
                        self.location.osm_index = self.goal_index
                        self.is_between = Vehicle.NOT_BETWEEN_TWO_INDEX
                        self.goal_index = -1
                        self.between_distance = 0
                        continue
                    else:
                        # 不能到目标节点
                        self.between_distance += average_speed
                        break
                else:
                    # 能回到当前节点
                    if current_trip_distance + self.between_distance - average_speed <= 0.01:
                        current_trip_distance += self.between_distance
                        self.is_between = Vehicle.NOT_BETWEEN_TWO_INDEX
                        self.goal_index = -1
                        self.between_distance = 0
                        continue
                    else:
                        # 不能回到当前节点
                        self.between_distance -= average_speed
                        break
            else:
                # 不在两个节点中间
                if current_trip_distance + two_location_distance - average_speed <= 0.01:  # 首先判断是否可以一分钟是否过剩
                    current_trip_distance += two_location_distance  # 更新当前行程的路径
                    self.location.osm_index = order_location.osm_index  # 更新车辆坐标
                    un_covered_location_index = index + 1  # 未完成订单坐标索引增加
                    belong_to_order = order_location.belong_to_order
                    if order_location.order_location_type == OrderLocation.PICK_UP_TYPE:  # 更新接乘客的形成路径
                        belong_to_order.pick_up_distance = self.trip_distance + current_trip_distance
                    if order_location.order_location_type == OrderLocation.DROP_OFF_TYPE:  # 更新车辆可用座位数量
                        self.n_seats += belong_to_order.n_riders
                    if current_trip_distance - average_speed >= -0.01:  # 行程刚好满足一分钟的情况
                        break
                else:  # 订单节点路长过大无法到达的情况
                    un_covered_location_index = index
                    current_index = self.location.osm_index
                    goal_index = order_location.osm_index
                    # 如果两个节点之间没有中间节点，且一分钟到不了后一个节点（或者最短路径后一个节点到达不了），记录中间位置
                    if goal_index == shortest_path[current_index][goal_index] or shortest_distance[current_index, goal_index] - average_speed > 0.01:
                        self.is_between = Vehicle.IS_BETWEEN_TWO_INDEX
                        self.between_distance += average_speed
                        self.goal_index = goal_index
                        break
                    while True:
                        next_index = shortest_path[current_index, goal_index]
                        partial_two_location_distance = shortest_distance[current_index, next_index]
                        if current_trip_distance + partial_two_location_distance - average_speed <= 0.01:
                            current_trip_distance += partial_two_location_distance
                            current_index = next_index
                        else:  # 选择最接近1min的路程
                            # current_trip_distance + partial_order_trip_distance - average_speed <
                            # average_speed - current_trip_distance
                            # 为了代码篇幅 我们采用的下面写法
                            if 2 * current_trip_distance + partial_two_location_distance < 2 * average_speed:
                                current_trip_distance += partial_two_location_distance
                                current_index = next_index
                            break

                    self.location.osm_index = current_index
                    break
        self.route_plan = self.route_plan[un_covered_location_index:]
        self.trip_distance += current_trip_distance
        if len(self.route_plan) != 0:
            self.status = Vehicle.HAVE_MISSION_STATUS
        else:
            self.status = Vehicle.WITHOUT_MISSION_STATUS

    # toString()
    def __repr__(self):
        return "id is {0} location = {1} available_seats = {2} trip_distance = {3} route_plan = {4}". \
            format(self.vehicle_id, self.location, self.n_seats, self.trip_distance, self.route_plan)

    def __hash__(self):
        return hash(self.vehicle_id)

    def __eq__(self, other):
        return self.vehicle_id == other.vehicle_id


if __name__ == '__main__':
    # TODO: 测试agent的功能
    l_m = {0:  (0.0, 0.0),  1: (0.0, 1.0), 2:  (0.0, 2.0), 3:  (0.0, 3.0),
           4:  (1.0, 0.0),  5: (1.0, 1.0), 6:  (1.0, 2.0), 7:  (1.0, 3.0),
           8:  (2.0, 0.0),  9: (2.0, 1.0), 10: (2.0, 2.0), 11: (2.0, 3.0),
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
    p0 = GeoLocation(0)
    p1 = OrderLocation(5, OrderLocation.PICK_UP_TYPE)
    p2 = OrderLocation(6, OrderLocation.DROP_OFF_TYPE)
    p3 = OrderLocation(8, OrderLocation.PICK_UP_TYPE)
    p4 = OrderLocation(14, OrderLocation.DROP_OFF_TYPE)
    p5 = OrderLocation(11, OrderLocation.PICK_UP_TYPE)
    p6 = OrderLocation(15, OrderLocation.DROP_OFF_TYPE)

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
    distance_o1_o2 = s_d[order1.end_location.osm_index][order2.start_location.osm_index]
    print("o2", distance_2)
    print("o1_o2", distance_o1_o2)

    # # vehicle_id: int, location: GeoLocation, n_seats: int, cost_per_km: float, status
    vehicle1 = Vehicle(0, p0, 4, 1.0, Vehicle.WITHOUT_MISSION_STATUS)
    rem_list = []
    rem_list.append(order1.start_location)
    rem_list.append(order2.start_location)
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
    # vehicle1.update_order_location(s_d, s_p)
    # print(vehicle1)