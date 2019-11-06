#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/29
import numpy as np
from typing import List
from typing import Dict
from typing import Tuple
from array import array
from setting import DISTANCE_EPS
from constant import INT_ZERO
from constant import FLOAT_ZERO
from constant import FIRST_INDEX
from env.location import GeoLocation
from env.location import VehicleLocation
from env.location import OrderLocation
from utility import equal

__all__ = ["_Graph"]


class _Graph:
    """
    路网环境
    """
    __slots__ = ["shortest_distance", "shortest_path", "access_index",
                 "adjacent_location_osm_index", "adjacent_location_driven_distance", "adjacent_location_goal_index",
                 "index2location", "index2osm_id", "graph"]

    def __init__(self, shortest_distance: np.ndarray, shortest_path: np.ndarray,
                 access_index: List[array],
                 adjacent_location_osm_index: List[array],
                 adjacent_location_driven_distance: List[array],
                 adjacent_location_goal_index: List[array],
                 index2location: Dict[int, Tuple[float, float]],
                 index2osm_id: Dict[int, int], graph):
        """
        :param shortest_distance: 两个节点最短路径距离矩阵
        :param shortest_path: 两个节点最短路径矩阵  shortest_path[i,j]->k 表示i到j的最短路径需要经过k
        :param access_index: 表示车辆在某一个节点上可以到达的节点
        :param adjacent_location_osm_index: 保存车辆下一个时间间隔可以到达的节点
        :param adjacent_location_driven_distance: 保存车辆下一个时间间隔可以到达的节点 还会多行驶的一段距离
        :param adjacent_location_goal_index: 保存车辆下一个时间间隔可以到达的节点 多行驶距离的朝向节点
        :param index2location: 用于与底层的数据进行转换对接，自己坐标的运动体系index->osm_id->(longitude, latitude)
        :param index2osm_id: 用于与底层的数据进行转换啊对接，自己坐标的运动体系index->osm_id
        :param graph: 真实的图
        ------
        注意：
        shortest_distance 用于查询任意两点之间的最短路径长度 单位长度m
        1. i==j, shortest_length[i,j] = 0;
        2. i不可以到达j, shortest_length[i, j] = np.inf

        shortest_path 用于记录两点按照最短路径走下一步会到哪个节点
        1. shortest_distance[i, j] == 0.0, shortest_path[i, j] = -1;
        2. shortest_distance[i, j] == np.inf, shortest_path[i, j] = -2;
        """
        self.graph = graph
        self.index2osm_id = index2osm_id
        self.shortest_distance = shortest_distance
        self.shortest_path = shortest_path
        self.access_index = access_index
        self.adjacent_location_osm_index = adjacent_location_osm_index
        self.adjacent_location_driven_distance = adjacent_location_driven_distance
        self.adjacent_location_goal_index = adjacent_location_goal_index
        self.index2location = index2location

    def __real_drive_to_target_index(self, vehicle_location: VehicleLocation, target_index: int, could_drive_distance: float):
        """
        模拟一个车辆真实得向一个可以到达得目标节点前进过程, 用于精细化调整
        :param vehicle_location:
        :param target_index:
        :param could_drive_distance:
        :return:
        """
        if vehicle_location.is_between:
            raise Exception("车辆不是固定在一个点上的无法继续进行后续的计算")
        vehicle_to_target_distance = FLOAT_ZERO
        while could_drive_distance > DISTANCE_EPS and not equal(could_drive_distance, DISTANCE_EPS):  # 需要行驶的距离已经过小了的情况
            goal_index = self.shortest_path[vehicle_location.osm_index, target_index]
            vehicle_to_goal_distance = self.shortest_distance[vehicle_location.osm_index, goal_index]
            if vehicle_to_goal_distance - could_drive_distance < DISTANCE_EPS or equal(vehicle_to_goal_distance-could_drive_distance, DISTANCE_EPS):
                could_drive_distance -= vehicle_to_goal_distance
                vehicle_to_target_distance += vehicle_to_goal_distance
            else:
                vehicle_location.set_location(vehicle_location.osm_index, could_drive_distance, goal_index)
                vehicle_to_target_distance += could_drive_distance
                break
            vehicle_location.osm_index = goal_index
        return vehicle_to_target_distance

    def __real_try_drive_to_target_index(self, vehicle_location: VehicleLocation, target_index: int, could_drive_distance: float):
        """
        模拟一个车辆真实得尝试向某一个可以到达的目标节点前进的过程
        :param vehicle_location:
        :param target_index:
        :param could_drive_distance:
        :return:
        ------
        注意：这个函数会修改vehicle_location的值
        """
        if vehicle_location.is_between:
            raise Exception("车辆不是固定在一个点上的无法继续进行后续的计算")
        vehicle_to_target_distance = self.shortest_distance[vehicle_location.osm_index, target_index]
        if vehicle_to_target_distance - could_drive_distance < DISTANCE_EPS or \
                equal(vehicle_to_target_distance - could_drive_distance, DISTANCE_EPS):  # 可以到target_index的情况
            vehicle_location.set_location(target_index)
            partial_drive_distance = vehicle_to_target_distance
        else:
            partial_drive_distance = self.__real_drive_to_target_index(vehicle_location, target_index, could_drive_distance)
        return partial_drive_distance

    def get_random_vehicle_locations(self, vehicle_number: int):
        """
        用于返回一个随机车辆位置
        :return:
        """
        locations_index = np.random.choice(list(self.index2location.keys()), vehicle_number)
        locations = [VehicleLocation(locations_index[i]) for i in range(vehicle_number)]
        return locations

    def get_shortest_distance(self, location1: GeoLocation, location2: GeoLocation) -> float:
        return self.shortest_distance[location1.osm_index, location2.osm_index]

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
        distance_a = self.shortest_distance[vehicle_location.osm_index, order_location.osm_index]  # 意义看文档
        distance_b = self.shortest_distance[vehicle_location.goal_index, order_location.osm_index]  # 意义看文档
        distance_c = self.shortest_distance[vehicle_location.osm_index, vehicle_location.goal_index]  # 意义看文档
        # 用于判断车是否从goal_index到order_location
        diff = (distance_c - vehicle_location.driven_distance + distance_b) - (distance_a + vehicle_location.driven_distance)
        flag1 = diff < FLOAT_ZERO or equal(diff, FLOAT_ZERO)
        # 用于判断是否可以反向行驶
        flag2 = self.shortest_distance[vehicle_location.goal_index, vehicle_location.osm_index] == np.inf
        return flag1 or flag2

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
        distance_a = self.shortest_distance[vehicle_location.osm_index, order_location.osm_index]  # 意义看文档
        if vehicle_location.is_between:
            if self.check_vehicle_on_road_or_not(vehicle_location, order_location):  # 车无法通过goal_index到达order_location
                distance_b = self.shortest_distance[vehicle_location.goal_index, order_location.osm_index]  # 意义看文档
                distance_c = self.shortest_distance[vehicle_location.osm_index, vehicle_location.goal_index]  # 意义看文档
                rest_pick_up_distance = distance_c - vehicle_location.driven_distance + distance_b
            else:
                rest_pick_up_distance = distance_a + vehicle_location.driven_distance
        else:
            rest_pick_up_distance = distance_a

        return rest_pick_up_distance

    def simulate_drive_on_route_plan(self, vehicle_location: VehicleLocation, new_route_plan: List[OrderLocation]):
        """
        一个模拟器，模拟车辆按照订单顺序行走，注意这个只是一个模拟器，不会修改车辆任何值，每一次yield 出当前行驶到订单位置，和行驶的距离
        :param vehicle_location: 车辆坐标
        :param new_route_plan: 路径规划
        ------
        注意：
        这个函数不可以修改 vehicle_location的值
        """
        pre_drive_distance = FLOAT_ZERO
        move_osm_index = vehicle_location.osm_index
        if vehicle_location.is_between:  # 如果车两个节点之间需要提前行走一段距离
            if self.check_vehicle_on_road_or_not(vehicle_location, new_route_plan[FIRST_INDEX]):  # 车无法通过goal_index到达order_location
                move_osm_index = vehicle_location.goal_index
                pre_drive_distance = self.shortest_distance[vehicle_location.osm_index, vehicle_location.goal_index] - \
                    vehicle_location.driven_distance
            else:
                pre_drive_distance = vehicle_location.driven_distance

        vehicle_to_order_distance = pre_drive_distance  # vehicle_to_order_distance表示每一次车辆到一个订单节点的距离，首先车辆到订单的距离加上这个预先行驶距离
        for order_location in new_route_plan:
            vehicle_to_order_distance += self.shortest_distance[move_osm_index, order_location.osm_index]
            yield order_location, vehicle_to_order_distance
            move_osm_index = order_location.osm_index
            vehicle_to_order_distance = FLOAT_ZERO

    def real_drive_on_random(self, vehicle_location: VehicleLocation, could_drive_distance: float):
        """
        :param vehicle_location:  车辆当前的位置
        :param could_drive_distance 车辆可以行驶的距离
        ------
        注意： 这个函数会修改vehicle_location的值!!!!!!
        在一个时间间隔内，车辆随机路网上行驶
        """
        this_time_drive_distance = FLOAT_ZERO
        if vehicle_location.is_between:  # 车辆处于两个节点之间
            # 首先车辆行驶两节点之间的距离
            vehicle_to_goal_distance = self.shortest_distance[vehicle_location.osm_index, vehicle_location.goal_index] - \
                                       vehicle_location.driven_distance

            if vehicle_to_goal_distance - could_drive_distance < DISTANCE_EPS or \
                    equal(vehicle_to_goal_distance - could_drive_distance, DISTANCE_EPS):  # 可以到goal_index上
                vehicle_location.set_location(vehicle_location.goal_index)  # 更新车辆位置
                this_time_drive_distance += vehicle_to_goal_distance   # 增加车辆行驶的距离
                could_drive_distance -= vehicle_to_goal_distance   # 减小车辆可以行驶的距离
            else:  # 不可以行驶动goal_index上
                vehicle_location.driven_distance += could_drive_distance  # 这样车辆还是会在两节点之间但是需要修改已经行驶的距离
                this_time_drive_distance += could_drive_distance  # 得到车辆当前行驶的距离
                could_drive_distance = FLOAT_ZERO

            # 如果车辆还有可行驶的距离探索可达节点
            if could_drive_distance > DISTANCE_EPS and not equal(could_drive_distance, DISTANCE_EPS) and \
                    len(self.access_index[vehicle_location.osm_index]) != INT_ZERO:
                target_index = np.random.choice(self.access_index[vehicle_location.osm_index])
                this_time_drive_distance += self.__real_try_drive_to_target_index(vehicle_location, target_index, could_drive_distance)
        else:
            if len(self.adjacent_location_osm_index[vehicle_location.osm_index]) == INT_ZERO:  # 一分钟到不了任何节点，随机选择一个可以到达节点作为目标节点前进
                if len(self.access_index[vehicle_location.osm_index]) == INT_ZERO:  # 车当前的节点不可以到任何节点，那么就凭空移动，帮助其摆脱陷阱
                    target_index = np.random.choice(range(len(self.shortest_distance)))
                    vehicle_location.set_location(target_index)  # 凭空迁移
                else:
                    target_index = np.random.choice(self.access_index[vehicle_location.osm_index])
                    this_time_drive_distance += self.__real_try_drive_to_target_index(vehicle_location, target_index, could_drive_distance)
            else:
                idx = np.random.choice(range(len(self.adjacent_location_osm_index[vehicle_location.osm_index])))
                osm_index = self.adjacent_location_osm_index[vehicle_location.osm_index][idx]
                driven_distance = self.adjacent_location_driven_distance[vehicle_location.osm_index][idx]
                goal_index = self.adjacent_location_goal_index[vehicle_location.osm_index][idx]
                vehicle_to_goal_distance = self.shortest_distance[vehicle_location.osm_index, osm_index]
                this_time_drive_distance += vehicle_to_goal_distance
                if osm_index != goal_index:
                    vehicle_location.set_location(osm_index, driven_distance, goal_index)
                    this_time_drive_distance += driven_distance
                else:
                    vehicle_location.set_location(osm_index)

        return this_time_drive_distance

    def real_drive_on_route_plan(self, vehicle_location: VehicleLocation, route_plan: List[OrderLocation], could_drive_distance: float):
        """
        在一个时间间隔内，车辆按照自己的路径规划进行行驶
        :param vehicle_location: 车俩当前的位置
        :param route_plan: 车辆当前的路径规划
        :param could_drive_distance: 车辆可以行驶的距离
        ------
        注意：
        这个函数会修改vehicle_location的值
        """
        if vehicle_location.is_between:  # 当前车辆在两点之间
            if self.check_vehicle_on_road_or_not(vehicle_location, route_plan[FIRST_INDEX]):  # 当前车辆需要向location.goal_index行驶
                vehicle_to_goal_distance = self.shortest_distance[vehicle_location.osm_index, vehicle_location.goal_index] - \
                                           vehicle_location.driven_distance
                # 判断是否可以到location.goal_index
                # 1. vehicle 到 goal_index 的距离远远小于should_drive_distance
                # 2. vehicle 到 goal_index 的距离只比should_drive_distance大DISTANCE_EPS
                if vehicle_to_goal_distance - could_drive_distance < DISTANCE_EPS or \
                        equal(vehicle_to_goal_distance - could_drive_distance, DISTANCE_EPS):
                    pre_drive_distance = vehicle_to_goal_distance
                    vehicle_location.set_location(vehicle_location.goal_index)  # 移动到对应的位置上
                else:
                    pre_drive_distance = could_drive_distance
                    vehicle_location.driven_distance += could_drive_distance  # 还在之前的两个节点之间只需要修改行驶距离就可以，只不过是向goal_index方向行驶
            else:
                # 判断是否可以回到上一个出发节点
                # 1. vehicle 到 vehicle_location 的距离远远小于should_drive_distance
                # 2. vehicle 到 vehicle_location 的距离只是比should_drive_distance大10m
                driven_distance = vehicle_location.driven_distance
                if driven_distance - could_drive_distance < DISTANCE_EPS or equal(driven_distance - could_drive_distance, DISTANCE_EPS):
                    pre_drive_distance = driven_distance
                    vehicle_location.reset()  # 又回到了osm_index上了
                else:
                    pre_drive_distance = could_drive_distance
                    vehicle_location.driven_distance -= could_drive_distance  # 还在之前的两个节点之间只需要修改行驶距离就可以，只不过是向osm_index方向行驶
        else:
            pre_drive_distance = FLOAT_ZERO

        could_drive_distance -= pre_drive_distance

        if could_drive_distance > DISTANCE_EPS and not equal(could_drive_distance, DISTANCE_EPS):  # 说明之前车辆的位置更新之后，需要行驶距离还没有用完
            for covered_index, order_location in enumerate(route_plan):  # 现在开始探索路线规划的各个节点
                vehicle_to_order_distance = self.get_shortest_distance(vehicle_location, order_location)
                if covered_index == FIRST_INDEX:
                    vehicle_to_order_distance += pre_drive_distance

                if vehicle_to_order_distance - could_drive_distance < DISTANCE_EPS or \
                        equal(vehicle_to_order_distance - could_drive_distance, DISTANCE_EPS):  # 当此订单节点是可以到达的情况
                    could_drive_distance -= vehicle_to_order_distance  # 更新当前车辆需要行驶的距离
                    vehicle_location.osm_index = order_location.osm_index  # 更新车辆坐标

                    yield True, covered_index, order_location, vehicle_to_order_distance

                    if could_drive_distance < DISTANCE_EPS or equal(could_drive_distance, DISTANCE_EPS):  # 需要行驶的距离已经过小了的情况下直接拉到对应的位置
                        break
                else:  # 订单节点路长过大无法到达的情况, 需要进行精细调整
                    target_index = order_location.osm_index
                    if covered_index == FIRST_INDEX:  # 如果是第一个订单就是不可达的那么要考虑之前行驶的距离
                        vehicle_to_order_distance = self.__real_drive_to_target_index(vehicle_location, target_index, could_drive_distance) + \
                                                    pre_drive_distance
                    else:
                        vehicle_to_order_distance = self.__real_drive_to_target_index(vehicle_location, target_index, could_drive_distance)
                    yield False, covered_index - 1, None, vehicle_to_order_distance
                    break
        else:
            yield False, -1, None, pre_drive_distance
