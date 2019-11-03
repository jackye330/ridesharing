#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/29
from typing import List, Dict, Tuple

import numpy as np

from setting import DISTANCE_EPS
from network.transport_network import GeoLocation, VehicleLocation, OrderLocation

__all__ = ["NetworkGraph"]


class NetworkGraph:
    __slots__ = ["shortest_distance", "shortest_path", "shortest_path_time_slot",
                 "shortest_path_driven_distance", "shortest_path_goal_index",
                 "index2location", "index2osm_id"]

    def __init__(self, shortest_distance: np.ndarray, shortest_path: np.ndarray, shortest_path_time_slot: np.ndarray,
                 shortest_path_driven_distance: np.ndarray, shortest_path_goal_index: np.ndarray,
                 index2location: Dict[int, Tuple[float, float]], index2osm_id: Dict[int, int]):
        """
        :param shortest_distance: 两个节点最短路径距离矩阵
        :param shortest_path: 两个节点最短路径矩阵  shortest_path[i,j]->k 表示i到j的最短路径需要经过k
        :param shortest_path_time_slot: 两个节点最短路矩阵 shortest_path_time_slot[i,j]->k 表示i到j的过程里面一个时间间隔需要经过k
        :param shortest_path_driven_distance: 在车辆按照最短路径行走下，经过一个时间间隔行驶多行驶超过一个节点的距离
        :param shortest_path_goal_index: 在车辆按照最短路径行走下，经过一个时间间隔多行驶一段距离的朝向节点
        :param index2location: 用于与底层的数据进行转换对接，自己坐标的运动体系index->osm_id->(longitude, latitude)
        :param index2osm_id: 用于与底层的数据进行转换对接，自己坐标系的索引转换index->osm_id
        ------
        注意：
        shortest_distance 用于查询任意两点之间的最短路径长度 单位长度m
        1. i==j, shortest_length[i,j] = 0;
        2. i不可以到达j, shortest_length[i, j] = np.inf

        shortest_path 用于记录两点按照最短路径走下一步会到哪个节点
        1. shortest_distance[i, j] == 0.0, shortest_path[i, j] = -1;
        2. shortest_distance[i, j] == np.inf, shortest_path[i, j] = -2;

        shortest_path_time_slot 用于车辆随机行走的查询经过一个时间间隔会到达的节点
        1. shortest_distance[i, j] 太小 （shortest_distance[i,j] << 一段时间行走的距离）, shortest_path_time_slot[i, j] = -1;
        2. shortest_distance[i, j] == np.inf, shortest_path_time_slot[i, j] = -2;
        3. 如果存在车辆一个时间间隔的路程刚好可以跨越两个半相邻的节点那么取最接近一个间隔路程的节点
        """
        self.shortest_distance = shortest_distance
        self.shortest_path = shortest_path
        self.shortest_path_time_slot = shortest_path_time_slot
        self.shortest_path_driven_distance = shortest_path_driven_distance
        self.shortest_path_goal_index = shortest_path_goal_index
        self.index2location = index2location
        self.index2osm_id = index2osm_id

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

    def real_vehicle_drive_on_route_plan(self, vehicle_location: VehicleLocation, vehicle_route_plan: List[OrderLocation],
                                         should_drive_distance: float):
        """
        在一个时间间隔内，车辆按照自己的路径规划进行行驶
        :param vehicle_location: 车俩当前的位置
        :param vehicle_route_plan: 车辆当前的路径规划
        :param should_drive_distance: 车俩需要行驶的距离
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
                if vehicle_to_goal_distance - should_drive_distance <= DISTANCE_EPS:
                    pre_drive_distance = vehicle_to_goal_distance
                    vehicle_location.osm_index = vehicle_location.goal_index
                    vehicle_location.reset()  # 由于不在两点之间了需要重置goal_index和相应的一些设置
                else:
                    pre_drive_distance = should_drive_distance
                    vehicle_location.driven_distance += should_drive_distance
            else:
                # 判断是否可以回到上一个出发节点
                # 1. vehicle 到 vehicle_location 的距离远远小于should_drive_distance
                # 2. vehicle 到 vehicle_location 的距离只是比should_drive_distance大10m
                if vehicle_location.driven_distance - should_drive_distance <= DISTANCE_EPS:
                    pre_drive_distance = vehicle_location.driven_distance
                    vehicle_location.reset()  # 由于不在两点之间了需要重置goal_index和相应的一些设置
                else:
                    pre_drive_distance = should_drive_distance
                    vehicle_location.driven_distance -= should_drive_distance
        else:
            pre_drive_distance = 0.0

        if not vehicle_location.is_between:  # 说明之前车辆的位置更新之后，需要行驶距离还没有用完
            for covered_index, order_location in enumerate(vehicle_route_plan):  # 现在开始探索路线规划的各个节点
                vehicle_to_order_distance = self.get_shortest_distance(vehicle_location, order_location)
                if covered_index == 0:
                    vehicle_to_order_distance += pre_drive_distance

                if vehicle_to_order_distance - should_drive_distance <= DISTANCE_EPS:  # 当此订单节点是可以到达的情况
                    should_drive_distance -= vehicle_to_order_distance  # 更新当前车辆需要行驶的距离
                    vehicle_location.osm_index = order_location.osm_index  # 更新车辆坐标

                    yield True, covered_index, order_location, vehicle_to_order_distance

                    if should_drive_distance <= DISTANCE_EPS:  # 需要行驶的距离已经过小了的情况下直接拉到对应的位置
                        break
                else:  # 订单节点路长过大无法到达的情况, 需要进行精细调整
                    vehicle_to_order_distance = 0.0
                    while True:
                        goal_index = self.shortest_path[vehicle_location.osm_index, order_location.osm_index]
                        partial_two_location_distance = self.shortest_distance[vehicle_location.osm_index,
                                                                               order_location.osm_index]
                        if partial_two_location_distance - should_drive_distance <= DISTANCE_EPS:
                            should_drive_distance -= partial_two_location_distance
                            vehicle_to_order_distance += partial_two_location_distance
                            vehicle_location.osm_index = goal_index
                            if should_drive_distance <= DISTANCE_EPS:  # 需要行驶的距离已经过小了的情况下直接拉到对应的位置
                                break
                        else:
                            vehicle_location.is_between = True
                            vehicle_location.goal_index = goal_index
                            vehicle_location.driven_distance = should_drive_distance
                            vehicle_to_order_distance += should_drive_distance
                            break
                    yield False, covered_index - 1, order_location, vehicle_to_order_distance
                    break
        else:
            yield False, -1, None, pre_drive_distance

    def real_vehicle_drive_on_random(self, vehicle_location: VehicleLocation, vehicle_route_plan: List[OrderLocation],
                                     should_drive_distance: float):
        """
        注意： 这个函数会修改vehicle_location的值!!!!!!
        在一个时间间隔内，车辆随机路网上行驶
        :param vehicle_location:
        :param vehicle_route_plan:
        :param should_drive_distance:
        :return:
        """
        pass


if __name__ == '__main__':
    pass
