#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/29
from typing import List
from typing import Dict
from typing import Tuple
from array import array
import numpy as np

from setting import DISTANCE_EPS
from network.transport_network.location import GeoLocation, VehicleLocation, OrderLocation

__all__ = ["Graph"]


class Graph:
    __slots__ = ["shortest_distance", "shortest_path", "access_index", "adjacent_index",
                 "adjacent_location_osm_index", "adjacent_location_driven_distance", "adjacent_location_goal_index",
                 "index2location", "index2osm_id"]

    def __init__(self, shortest_distance: np.ndarray, shortest_path: np.ndarray,
                 access_index: List[array],
                 adjacent_index: List[array],
                 adjacent_location_osm_index: List[array],
                 adjacent_location_driven_distance: List[array],
                 adjacent_location_goal_index: List[array],
                 index2location: Dict[int, Tuple[float, float]]):
        """
        :param shortest_distance: 两个节点最短路径距离矩阵
        :param shortest_path: 两个节点最短路径矩阵  shortest_path[i,j]->k 表示i到j的最短路径需要经过k
        :param access_index: 表示车辆在某一个节点上可以到达的节点
        :param adjacent_index: 保存一个节点相邻的节点
        :param adjacent_location_osm_index: 保存车辆下一个时间间隔可以到达的节点
        :param adjacent_location_driven_distance: 保存车辆下一个时间间隔可以到达的节点 还会多行驶的一段距离
        :param adjacent_location_goal_index: 保存车辆下一个时间间隔可以到达的节点 多行驶距离的朝向节点
        :param index2location: 用于与底层的数据进行转换对接，自己坐标的运动体系index->osm_id->(longitude, latitude)
        ------
        注意：
        shortest_distance 用于查询任意两点之间的最短路径长度 单位长度m
        1. i==j, shortest_length[i,j] = 0;
        2. i不可以到达j, shortest_length[i, j] = np.inf

        shortest_path 用于记录两点按照最短路径走下一步会到哪个节点
        1. shortest_distance[i, j] == 0.0, shortest_path[i, j] = -1;
        2. shortest_distance[i, j] == np.inf, shortest_path[i, j] = -2;
        """
        self.shortest_distance = shortest_distance
        self.shortest_path = shortest_path
        self.access_index = access_index
        self.adjacent_index = adjacent_index
        self.adjacent_location_osm_index = adjacent_location_osm_index
        self.adjacent_location_driven_distance = adjacent_location_driven_distance
        self.adjacent_location_goal_index = adjacent_location_goal_index
        self.index2location = index2location

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
        flag1 = distance_c - vehicle_location.driven_distance + distance_b < distance_a + vehicle_location.driven_distance
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

    def compute_vehicle_pre_travel_distance(self, vehicle_location: VehicleLocation, order_location: OrderLocation) -> float:
        """"
        如果现在两个节点之间车辆需要预先行驶一段距离，我们计算这个距离
        情景是下面的情况:
                            / order_location \
                    distance_a             distance_b
                        /     distance_c       \
        location.osm_index-----vehicle------location.goal_index
        :param vehicle_location: 车辆位置
        :param order_location: 订单的位置
        :return: pre_drive_distance: 接乘客还需要行走的距离
        ------
        注意：
        这个函数不可以修改 vehicle_location的值
        """
        if vehicle_location.is_between:  # 如果车两个节点之间需要提前行走一段距离
            if self.check_vehicle_on_road_or_not(vehicle_location, order_location):  # 车无法通过goal_index到达order_location
                two_location_distance = self.shortest_distance[vehicle_location.osm_index, vehicle_location.goal_index]
                pre_drive_distance = two_location_distance - vehicle_location.driven_distance  # 计算两个节点剩下的路程
            else:
                pre_drive_distance = vehicle_location.driven_distance
        else:
            pre_drive_distance = 0.0
        return pre_drive_distance

    def simulate_vehicle_drive_on_route_plan(self, vehicle_location: VehicleLocation, new_route_plan: List[OrderLocation]):
        """
        一个模拟器，模拟车辆按照订单顺序行走，注意这个只是一个模拟器，不会修改车辆任何值，每一次yield 出当前行驶到订单位置，和行驶的距离
        :param vehicle_location: 车辆坐标
        :param new_route_plan: 路径规划
        ------
        注意：
        这个函数不可以修改 vehicle_location的值
        """
        pre_drive_distance = 0.0
        move_osm_index = vehicle_location.osm_index
        if vehicle_location.is_between:  # 如果车两个节点之间需要提前行走一段距离
            if self.check_vehicle_on_road_or_not(vehicle_location, new_route_plan[0]):  # 车无法通过goal_index到达order_location
                move_osm_index = vehicle_location.goal_index
                pre_drive_distance = self.shortest_distance[vehicle_location.osm_index, vehicle_location.goal_index]
                pre_drive_distance -= vehicle_location.driven_distance
            else:
                pre_drive_distance = vehicle_location.driven_distance

        vehicle_order_distance = pre_drive_distance
        for order_location in new_route_plan:
            vehicle_order_distance += self.shortest_distance[move_osm_index, order_location.osm_index]
            yield order_location, vehicle_order_distance
            move_osm_index = order_location.osm_index
            vehicle_order_distance = 0.0

    def real_vehicle_drive_on_random(self, vehicle_location: VehicleLocation, could_drive_distance: float):
        """
        :param vehicle_location:  车辆当前的位置
        :param could_drive_distance:  一个时间间隔可以行驶的距离
        ------
        注意： 这个函数会修改vehicle_location的值!!!!!!
        在一个时间间隔内，车辆随机路网上行驶
        """
        if vehicle_location.driven_distance < 0:
            vehicle_location.reset()

        real_drive_distance = 0.0
        if vehicle_location.is_between:  # 车辆处于两个节点之间
            vehicle_to_goal_distance = self.shortest_distance[vehicle_location.osm_index, vehicle_location.goal_index]
            vehicle_to_goal_distance -= vehicle_location.driven_distance

            if vehicle_to_goal_distance - could_drive_distance <= DISTANCE_EPS:  # 可以到goal_index上
                vehicle_location.osm_index = vehicle_location.goal_index  # 更新车辆的位置
                vehicle_location.reset()
                real_drive_distance += vehicle_to_goal_distance  # 得到车辆当前行驶的距离
                could_drive_distance -= vehicle_to_goal_distance
            else:  # 不可以行驶动goal_index上
                real_drive_distance += could_drive_distance  # 得到车辆当前行驶的距离
                could_drive_distance = 0.0

            if could_drive_distance > DISTANCE_EPS and len(self.access_index[vehicle_location.osm_index]) != 0:  # 要还有可用的行驶距离并且有可达的下一个节点
                target_index = np.random.choice(self.adjacent_location_osm_index[vehicle_location.osm_index])  # 随机选择一个节点作为前进的目标

                if self.shortest_distance[vehicle_location.osm_index, target_index] - could_drive_distance <= DISTANCE_EPS:  # 可以到target_index的情况
                    vehicle_location.osm_index = target_index
                    vehicle_location.reset()
                    real_drive_distance += self.shortest_distance[vehicle_location.osm_index, target_index]
                    could_drive_distance -= self.shortest_distance[vehicle_location.osm_index, target_index]
                else:
                    while could_drive_distance > DISTANCE_EPS:
                        goal_index = self.shortest_path[vehicle_location.osm_index, target_index]
                        two_index_distance = self.shortest_distance[vehicle_location.osm_index, goal_index]
                        if two_index_distance - could_drive_distance <= DISTANCE_EPS:
                            real_drive_distance += two_index_distance
                            could_drive_distance -= two_index_distance
                        else:
                            real_drive_distance += could_drive_distance
                            vehicle_location.is_between = True
                            vehicle_location.driven_distance = could_drive_distance
                            vehicle_location.goal_index = goal_index
                            break
                        vehicle_location.osm_index = goal_index

        else:
            if len(self.adjacent_location_osm_index[vehicle_location.osm_index]) == 0:  # 在规定的时间到不了任何节点到不了任何节点
                # 一分钟到不了任何节点，用相邻节点的方法，记录中间位置
                if len(self.adjacent_index[vehicle_location.osm_index]) == 0:  # 没有相邻节点，将车凭空移动到其他地方
                    vehicle_location.osm_index = np.random.choice(range(len(self.shortest_path[vehicle_location.osm_index, :]))) # 随机选择一个点转移
                else:  # 随便选择一个相邻的节点
                    next_index = np.random.choice(self.adjacent_index[vehicle_location.osm_index])
                    real_drive_distance += self.shortest_distance[vehicle_location.osm_index, next_index]
                    vehicle_location.osm_index = next_index
                    vehicle_location.reset()
            else:
                idx = np.random.choice(range(len(self.adjacent_location_osm_index[vehicle_location.osm_index])))
                osm_index = self.adjacent_location_osm_index[vehicle_location.osm_index][idx]
                driven_distance = self.adjacent_location_driven_distance[vehicle_location.osm_index][idx]
                goal_index = self.adjacent_location_goal_index[vehicle_location.osm_index][idx]

                real_drive_distance += self.shortest_distance[vehicle_location.osm_index, osm_index]  # 一定要先处理
                vehicle_location.osm_index = osm_index
                if osm_index != goal_index:
                    vehicle_location.is_between = True
                    vehicle_location.driven_distance = driven_distance
                    vehicle_location.goal_index = goal_index
                    real_drive_distance += vehicle_location.driven_distance
                else:
                    vehicle_location.reset()

        return real_drive_distance

    def real_vehicle_drive_on_route_plan(self, vehicle_location: VehicleLocation, vehicle_route_plan: List[OrderLocation],
                                         could_drive_distance: float):
        """
        在一个时间间隔内，车辆按照自己的路径规划进行行驶
        :param vehicle_location: 车俩当前的位置
        :param vehicle_route_plan: 车辆当前的路径规划
        :param could_drive_distance: 车俩可以行驶的距离
        ------
        注意：
        这个函数会修改vehicle_location的值
        """
        if vehicle_location.driven_distance < 0:
            vehicle_location.reset()

        if vehicle_location.is_between:  # 当前车辆在两点之间
            if self.check_vehicle_on_road_or_not(vehicle_location, vehicle_route_plan[0]):  # 当前车辆需要向location.goal_index行驶
                vehicle_to_goal_distance = self.shortest_distance[vehicle_location.osm_index, vehicle_location.goal_index]
                vehicle_to_goal_distance -= vehicle_location.driven_distance
                # 判断是否可以到location.goal_index
                # 1. vehicle 到 goal_index 的距离远远小于should_drive_distance
                # 2. vehicle 到 goal_index 的距离只比should_drive_distance大DISTANCE_EPS
                if vehicle_to_goal_distance - could_drive_distance <= DISTANCE_EPS:
                    pre_drive_distance = vehicle_to_goal_distance
                    vehicle_location.osm_index = vehicle_location.goal_index
                    vehicle_location.reset()  # 由于不在两点之间了需要重置goal_index和相应的一些设置
                else:
                    pre_drive_distance = could_drive_distance
                    vehicle_location.driven_distance += could_drive_distance
            else:
                # 判断是否可以回到上一个出发节点
                # 1. vehicle 到 vehicle_location 的距离远远小于should_drive_distance
                # 2. vehicle 到 vehicle_location 的距离只是比should_drive_distance大10m
                if vehicle_location.driven_distance - could_drive_distance <= DISTANCE_EPS:
                    pre_drive_distance = vehicle_location.driven_distance
                    vehicle_location.reset()  # 由于不在两点之间了需要重置goal_index和相应的一些设置
                else:
                    pre_drive_distance = could_drive_distance
                    vehicle_location.driven_distance -= could_drive_distance
        else:
            pre_drive_distance = 0.0

        if not vehicle_location.is_between:  # 说明之前车辆的位置更新之后，需要行驶距离还没有用完
            for covered_index, order_location in enumerate(vehicle_route_plan):  # 现在开始探索路线规划的各个节点
                vehicle_to_order_distance = self.get_shortest_distance(vehicle_location, order_location)
                if covered_index == 0:
                    vehicle_to_order_distance += pre_drive_distance

                if vehicle_to_order_distance - could_drive_distance <= DISTANCE_EPS:  # 当此订单节点是可以到达的情况
                    could_drive_distance -= vehicle_to_order_distance  # 更新当前车辆需要行驶的距离
                    vehicle_location.osm_index = order_location.osm_index  # 更新车辆坐标

                    yield True, covered_index, order_location, vehicle_to_order_distance

                    if could_drive_distance <= DISTANCE_EPS:  # 需要行驶的距离已经过小了的情况下直接拉到对应的位置
                        break
                else:  # 订单节点路长过大无法到达的情况, 需要进行精细调整
                    vehicle_to_order_distance = 0.0
                    while could_drive_distance > DISTANCE_EPS:   # 需要行驶的距离已经过小了的情况
                        goal_index = self.shortest_path[vehicle_location.osm_index, order_location.osm_index]
                        partial_two_location_distance = self.shortest_distance[vehicle_location.osm_index, order_location.osm_index]
                        if partial_two_location_distance - could_drive_distance <= DISTANCE_EPS:
                            could_drive_distance -= partial_two_location_distance
                            vehicle_to_order_distance += partial_two_location_distance
                        else:
                            vehicle_location.is_between = True
                            vehicle_location.driven_distance = could_drive_distance
                            vehicle_location.goal_index = goal_index
                            vehicle_to_order_distance += could_drive_distance
                            break
                        vehicle_location.osm_index = goal_index
                    yield False, covered_index - 1, order_location, vehicle_to_order_distance
                    break
        else:
            yield False, -1, None, pre_drive_distance


if __name__ == '__main__':
    pass
