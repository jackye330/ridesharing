#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/3
from typing import List

import numpy as np
from utility import equal
from constant import FIRST_INDEX
from constant import FLOAT_ZERO
from constant import INT_ZERO
from setting import DISTANCE_EPS
from env.location import GeoLocation
from env.location import OrderLocation
from env.location import VehicleLocation

__all__ = ["_Graph"]


class _Graph:
    """
    网格环境
    """
    __slots__ = ["graph_size", "grid_size", "x_list", "y_list", "directions"]

    def __init__(self, graph_size: int, grid_size: float):
        """
        |----|
        |    | grid_size
        |----|
        :param graph_size: 表示网格横向，纵向得数目
        :param grid_size: 表示每一个网络方格的大小
        """
        self.graph_size = graph_size
        self.grid_size = grid_size
        self.x_list = [i for i in range(graph_size + 1)]
        self.y_list = [j for j in range(graph_size + 1)]
        self.directions = [(1, INT_ZERO),
                           (INT_ZERO, 1),
                           (-1, INT_ZERO),
                           (INT_ZERO, -1)]

    def __convert_index_to_xy(self, osm_index: int):
        return osm_index // self.graph_size, osm_index % self.graph_size

    def __convert_xy_to_index(self, x: int, y: int):
        return x * self.graph_size + y

    def __get_shortest_distance_by_xy(self, x1: int, y1: int, x2: int, y2: int):
        return (np.abs(x1 - x2) + np.abs(y1 - y2)) * self.grid_size

    def __get_shortest_distance_by_osm_index(self, osm_index1: int, osm_index2: int):
        x1, y1 = self.__convert_index_to_xy(osm_index1)
        x2, y2 = self.__convert_index_to_xy(osm_index2)
        return self.__get_shortest_distance_by_xy(x1, y1, x2, y2)

    def __get_best_next_xy(self, now_x: int, now_y: int, target_x: int, target_y: int):
        """
        获取从（now_x, now_y） 到 （goal_x, target_y） 最优下一点最优节点
        :param now_x: 当前节点x
        :param now_y: 当前节点y
        :param target_x: 目标节点x
        :param target_y: 目标节点y
        :return:
        """
        next_xy_lists = [(now_x + direction[0], now_y + direction[1]) for direction in self.directions
                         if 0 <= now_x + direction[0] <= self.graph_size and 0 <= now_y + direction[1] <= self.graph_size]
        next_xy_lists = [(next_x, next_y, self.__get_shortest_distance_by_xy(next_x, next_y, target_x, target_y))
                         for next_x, next_y in next_xy_lists]
        next_xy_lists.sort(key=lambda x: x[2])
        return next_xy_lists[0][0], next_xy_lists[0][1]

    def __real_drive_to_target_index(self, vehicle_location: VehicleLocation, target_index: int, could_drive_distance: float):
        """
        模拟一个车辆真实得向一个可以到达得目标节点前进过程, 用于精细化调整
        :param vehicle_location:
        :param target_index:
        :param could_drive_distance:
        :return:
        """
        if vehicle_location.is_between:
            # vehicle_location.reset()
            raise Exception("车辆不是固定在一个点上的无法继续进行后续的计算")
        vehicle_to_target_distance = FLOAT_ZERO
        now_x, now_y = self.__convert_index_to_xy(vehicle_location.osm_index)
        target_x, target_y = self.__convert_index_to_xy(target_index)
        while could_drive_distance > DISTANCE_EPS and not equal(could_drive_distance, DISTANCE_EPS):  # 需要行驶的距离已经过小了的情况
            goal_x, goal_y = self.__get_best_next_xy(now_x, now_y, target_x, target_y)
            vehicle_to_goal_distance = self.grid_size  # 因为只是移动一个格子, 所以只是移动一个小格子
            if vehicle_to_goal_distance - could_drive_distance < DISTANCE_EPS or \
                    equal(vehicle_to_goal_distance - could_drive_distance, DISTANCE_EPS):
                could_drive_distance -= vehicle_to_goal_distance
                vehicle_to_target_distance += vehicle_to_goal_distance
            else:
                now_index = self.__convert_xy_to_index(now_x, now_y)
                goal_index = self.__convert_xy_to_index(goal_x, goal_y)
                vehicle_location.set_location(now_index, could_drive_distance, goal_index)
                vehicle_to_target_distance += could_drive_distance
                break
            now_x, now_y = goal_x, goal_y
        else:
            now_index = self.__convert_xy_to_index(now_x, now_y)  # 正常退出情况需要更新此时车辆的位置
            vehicle_location.set_location(now_index)
        return vehicle_to_target_distance

    def __real_try_drive_to_target_index(self, vehicle_location: VehicleLocation, target_index: int, could_drive_distance: float):
        """
        模拟一个车辆真实得尝试向某一个可以到达的目标节点前进的过程
        :param vehicle_location:
        :param target_index:
        :param could_drive_distance:
        :return:
        ------
        注意：这个函数会修改vehicle_location的值, 确保车辆一定是在一个点上的，而不是在两个节点之间
        """
        if vehicle_location.is_between:
            raise Exception("车辆不是固定在一个点上的无法继续进行后续的计算")
        vehicle_to_target_distance = self.__get_shortest_distance_by_osm_index(vehicle_location.osm_index, target_index)
        if vehicle_to_target_distance - could_drive_distance < DISTANCE_EPS or \
                equal(vehicle_to_target_distance - could_drive_distance, DISTANCE_EPS):  # 可以到target_index的情况
            vehicle_location.set_location(target_index)
            partial_drive_distance = vehicle_to_target_distance
        else:
            partial_drive_distance = self.__real_drive_to_target_index(vehicle_location, target_index, could_drive_distance)
        return partial_drive_distance

    def get_random_vehicle_locations(self, vehicle_number: int):
        """
        用于返回一些随机车辆位置
        :return:
        """
        xs = np.random.choice(self.x_list, vehicle_number)
        ys = np.random.choice(self.y_list, vehicle_number)
        locations_index = [self.__convert_xy_to_index(xs[i], ys[i]) for i in range(vehicle_number)]
        locations = [VehicleLocation(locations_index[i]) for i in range(vehicle_number)]
        return locations

    def get_shortest_distance(self, location1: GeoLocation, location2: GeoLocation) -> float:
        return self.__get_shortest_distance_by_osm_index(location1.osm_index, location2.osm_index)

    def check_vehicle_on_road_or_not(self, vehicle_location: VehicleLocation, order_location: OrderLocation) -> bool:
        """
        当车辆正在两个节点之间的时候，判断车辆是否经过vehicle_location的目标节点到订单节点
        情景是下面的情况：
                            / order_location \
                    distance_a             distance_b
                        /     distance_c       \
        location.osm_index-----vehicle------location.goal_index

        如果是 distance_c - self.between_distance + distance_b < distance_a + self.between_distance 或者
        不可以从goal_index 到 location.osm_index 或者
        那么返回 true
        :param vehicle_location: 车辆位置
        :param order_location: 订单的位置
        :return 返回一个bool值 为真表示可以从goal_index到达目标节点，而不可以则要从osm_index到达目标节点
        ------
        注意：
        这个函数不可以修改 vehicle_location的值
        """
        distance_a = self.__get_shortest_distance_by_osm_index(vehicle_location.osm_index, order_location.osm_index)  # 意义看文档
        distance_b = self.__get_shortest_distance_by_osm_index(vehicle_location.goal_index, order_location.osm_index)  # 意义看文档
        distance_c = self.__get_shortest_distance_by_osm_index(vehicle_location.osm_index, vehicle_location.goal_index)  # 意义看文档
        # 用于判断车是否从goal_index到order_location
        diff = (distance_c - vehicle_location.driven_distance + distance_b) - (distance_a + vehicle_location.driven_distance)
        return diff < FLOAT_ZERO or equal(diff, FLOAT_ZERO)

    def compute_vehicle_rest_pick_up_distance(self, vehicle_location: VehicleLocation, order_location: OrderLocation) -> float:
        """
        计算车辆从当前位置去某一个订单位置需要行驶的距离
        情景是下面的情况:
                            / order_location \
                    distance_a             distance_b
                        /     distance_c       \
        location.osm_index-----vehicle------location.goal_index
        :param vehicle_location: 车辆位置
        :param order_location: 订单的位置
        :return: rest_pick_up_distance: 接乘客还需要行走的距离
        ------
        注意：
        这个函数不可以修改 vehicle_location的值
        """
        distance_a = self.__get_shortest_distance_by_osm_index(vehicle_location.osm_index, order_location.osm_index)  # 意义看文档
        if vehicle_location.is_between:
            if self.check_vehicle_on_road_or_not(vehicle_location, order_location):  # 车无法通过goal_index到达order_location
                distance_b = self.__get_shortest_distance_by_osm_index(vehicle_location.goal_index, order_location.osm_index)  # 意义看文档
                distance_c = self.__get_shortest_distance_by_osm_index(vehicle_location.osm_index, vehicle_location.goal_index)  # 意义看文档
                rest_pick_up_distance = distance_c - vehicle_location.driven_distance + distance_b
            else:
                rest_pick_up_distance = distance_a + vehicle_location.driven_distance
        else:
            rest_pick_up_distance = distance_a

        return rest_pick_up_distance

    def simulate_drive_on_route_plan(self, vehicle_location: VehicleLocation, route_plan: List[OrderLocation]):
        """
        一个模拟器，模拟车辆按照订单顺序行走，注意这个只是一个模拟器，不会修改车辆任何值，每一次yield 出当前行驶到订单位置，和行驶的距离
        :param vehicle_location: 车辆坐标
        :param route_plan: 路径规划
        ------
        注意：
        这个函数不可以修改 vehicle_location的值
        """

        move_osm_index = vehicle_location.osm_index
        if vehicle_location.is_between:  # 如果车两个节点之间需要提前行走一段距离
            if self.check_vehicle_on_road_or_not(vehicle_location, route_plan[FIRST_INDEX]):  # 车无法通过goal_index到达order_location
                move_osm_index = vehicle_location.goal_index
                pre_drive_distance = self.__get_shortest_distance_by_osm_index(vehicle_location.osm_index, vehicle_location.goal_index)
                pre_drive_distance -= vehicle_location.driven_distance
            else:
                pre_drive_distance = vehicle_location.driven_distance
        else:
            pre_drive_distance = FLOAT_ZERO

        vehicle_to_order_distance = pre_drive_distance  # 首先车辆到订单的距离加上这个预先行驶距离
        for order_location in route_plan:
            vehicle_to_order_distance += self.__get_shortest_distance_by_osm_index(move_osm_index, order_location.osm_index)
            yield order_location, vehicle_to_order_distance
            move_osm_index = order_location.osm_index
            vehicle_to_order_distance = FLOAT_ZERO

    def real_drive_on_random(self, vehicle_location: VehicleLocation, could_drive_distance: float):  # TODO 需要进行改造修改
        """
        :param vehicle_location:  车辆当前的位置
        :param could_drive_distance: 车辆可行行驶的距离
        ------
        注意： 这个函数会修改vehicle_location的值!!!!!!
        在一个时间间隔内，车辆随机路网上行驶
        """
        time_slot_drive_distance = FLOAT_ZERO
        if vehicle_location.is_between:  # 车辆处于两个节点之间
            vehicle_to_goal_distance = self.__get_shortest_distance_by_osm_index(vehicle_location.osm_index, vehicle_location.goal_index) - \
                                       vehicle_location.driven_distance

            if vehicle_to_goal_distance - could_drive_distance < DISTANCE_EPS or equal(vehicle_to_goal_distance - could_drive_distance, DISTANCE_EPS):
                # 当可以将车移动到goal_index上的时候
                vehicle_location.set_location(vehicle_location.goal_index)
                time_slot_drive_distance += vehicle_to_goal_distance  # 得到车辆当前行驶的距离
                could_drive_distance -= vehicle_to_goal_distance
            else:  # 不可以行驶动goal_index上
                vehicle_location.driven_distance += could_drive_distance  # 这样车辆还是会在两节点之间但是需要修改已经行驶的距离
                time_slot_drive_distance += could_drive_distance  # 得到车辆当前行驶的距离
                could_drive_distance = FLOAT_ZERO

        if could_drive_distance > DISTANCE_EPS and not equal(could_drive_distance, DISTANCE_EPS):  # 还可以继续行驶
            target_x = np.random.choice(self.x_list)
            target_y = np.random.choice(self.y_list)
            target_index = self.__convert_xy_to_index(target_x, target_y)
            if target_index != vehicle_location.osm_index:
                time_slot_drive_distance += self.__real_try_drive_to_target_index(vehicle_location, target_index, could_drive_distance)
        return time_slot_drive_distance

    def real_drive_on_route_plan(self, vehicle_location: VehicleLocation, route_plan: List[OrderLocation], could_drive_distance: float):
        """
        在一个时间间隔内，车辆按照自己的路径规划进行行驶
        :param vehicle_location: 车俩当前的位置
        :param route_plan: 车辆当前的路径规划
        :param could_drive_distance: 车辆可行行驶的距离
        ------
        注意：
        这个函数会修改vehicle_location的值
        """
        if vehicle_location.is_between:  # 当前车辆在两点之间
            if self.check_vehicle_on_road_or_not(vehicle_location, route_plan[FIRST_INDEX]):  # 当前车辆需要向location.goal_index行驶
                vehicle_to_goal_distance = self.__get_shortest_distance_by_osm_index(vehicle_location.osm_index, vehicle_location.goal_index) - \
                                           vehicle_location.driven_distance
                # 判断是否可以到location.goal_index
                # 1. vehicle 到 goal_index 的距离远远小于should_drive_distance
                # 2. vehicle 到 goal_index 的距离只比should_drive_distance大DISTANCE_EPS
                if vehicle_to_goal_distance - could_drive_distance < DISTANCE_EPS or \
                        equal(vehicle_to_goal_distance - could_drive_distance, DISTANCE_EPS):
                    pre_drive_distance = vehicle_to_goal_distance
                    vehicle_location.set_location(vehicle_location.goal_index)  # 由于不在两点之间了需要重置goal_index和相应的一些设置
                else:
                    pre_drive_distance = could_drive_distance
                    vehicle_location.driven_distance += could_drive_distance  # 还在之前的两个节点之间只需要修改行驶距离就可以，只不过是向goal_index方向行驶
            else:
                # 判断是否可以回到上一个出发节点
                # 1. vehicle 到 vehicle_location 的距离远远小于should_drive_distance
                # 2. vehicle 到 vehicle_location 的距离只是比should_drive_distance大10m
                driven_distance = vehicle_location.driven_distance
                if driven_distance - could_drive_distance < DISTANCE_EPS or equal(driven_distance - could_drive_distance, DISTANCE_EPS):
                    pre_drive_distance = vehicle_location.driven_distance
                    vehicle_location.reset()  # 又回到osm_index上
                else:
                    pre_drive_distance = could_drive_distance
                    vehicle_location.driven_distance -= could_drive_distance  # 还在之前的两个节点之间只需要修改行驶距离就可以，只不过是向osm_index
        else:
            pre_drive_distance = FLOAT_ZERO

        could_drive_distance -= pre_drive_distance  # 减少预先行驶的距离

        if could_drive_distance > DISTANCE_EPS and not equal(could_drive_distance, DISTANCE_EPS):  # 说明之前车辆的位置更新之后，需要行驶距离还没有用完
            for covered_index, order_location in enumerate(route_plan):  # 现在开始探索路线规划的各个节点
                vehicle_to_order_distance = self.get_shortest_distance(vehicle_location, order_location)

                if vehicle_to_order_distance - could_drive_distance < DISTANCE_EPS or \
                        equal(vehicle_to_order_distance - could_drive_distance, DISTANCE_EPS):  # 当此订单节点是可以到达的情况
                    could_drive_distance -= vehicle_to_order_distance  # 更新当前车辆需要行驶的距离
                    vehicle_location.osm_index = order_location.osm_index  # 更新车辆坐标

                    if covered_index == FIRST_INDEX:
                        vehicle_to_order_distance += pre_drive_distance
                    yield True, covered_index, order_location, vehicle_to_order_distance

                    if could_drive_distance < DISTANCE_EPS or equal(could_drive_distance, DISTANCE_EPS):  # 需要行驶的距离已经过小了的情况下直接拉到对应的位置
                        break
                else:  # 订单节点路长过大无法到达的情况, 需要进行精细调整
                    target_index = order_location.osm_index
                    vehicle_to_order_distance = self.__real_drive_to_target_index(vehicle_location, target_index, could_drive_distance)
                    yield False, covered_index - 1, order_location, vehicle_to_order_distance
                    break
        else:
            yield False, -1, None, pre_drive_distance
