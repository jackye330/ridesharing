#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/7
from env.location import VehicleLocation
from env.location import OrderLocation
from typing import List, NoReturn
from array import array
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Tuple
import numpy as np

from constant import FLOAT_ZERO
from constant import INT_ZERO
from env.location import OrderLocation
from env.location import VehicleLocation
from setting import DISTANCE_EPS
from utility import equal

__all__ = ["GridGraph", "RealGraph"]


class BaseGraph:
    """
    我的交通路网图的接口
    """
    __slots__ = []

    def get_shortest_distance_by_osm_index(self, osm_index1: int, osm_index2: int) -> float:
        raise NotImplementedError

    def check_vehicle_on_road_or_not(self, vehicle_location: VehicleLocation, order_location: OrderLocation):
        raise NotImplementedError

    def get_random_vehicle_locations(self, vehicle_number: int) -> List[VehicleLocation]:
        raise NotImplementedError

    def move_to_target_index(self, vehicle_location: VehicleLocation, target_index: int, could_drive_distance: float) -> NoReturn:
        raise NotImplementedError

    def random_move(self, vehicle_location: VehicleLocation, could_drive_distance: float) -> float:
        raise NotImplementedError


class RealGraph(BaseGraph):
    """
    实际的交通路网图
    """
    __slots__ = ["shortest_distance", "shortest_path", "access_index",
                 "adjacent_location_osm_index", "adjacent_location_driven_distance", "adjacent_location_goal_index",
                 "index2location", "index2osm_id", "raw_graph", "index_set"]

    def __init__(self, shortest_distance: np.ndarray, shortest_path: np.ndarray,
                 access_index: List[array],
                 adjacent_location_osm_index: List[array],
                 adjacent_location_driven_distance: List[array],
                 adjacent_location_goal_index: List[array],
                 index2location: Dict[int, Tuple[float, float]],
                 index2osm_id: Dict[int, int], raw_graph):
        """
        :param shortest_distance: 两个节点最短路径距离矩阵
        :param shortest_path: 两个节点最短路径矩阵  shortest_path[i,j]->k 表示i到j的最短路径需要经过k
        :param access_index: 表示车辆在某一个节点上可以到达的节点
        :param adjacent_location_osm_index: 保存车辆下一个时间间隔可以到达的节点
        :param adjacent_location_driven_distance: 保存车辆下一个时间间隔可以到达的节点 还会多行驶的一段距离
        :param adjacent_location_goal_index: 保存车辆下一个时间间隔可以到达的节点 多行驶距离的朝向节点
        :param index2location: 用于与底层的数据进行转换对接，自己坐标的运动体系index->osm_id->(longitude, latitude)
        :param index2osm_id: 用于与底层的数据进行转换啊对接，自己坐标的运动体系index->osm_id
        :param raw_graph: 真实的图
        ------
        注意：
        shortest_distance 用于查询任意两点之间的最短路径长度 单位长度m
        1. i==j, shortest_length[i,j] = 0;
        2. i不可以到达j, shortest_length[i, j] = np.inf

        shortest_path 用于记录两点按照最短路径走下一步会到哪个节点
        1. shortest_distance[i, j] == 0.0, shortest_path[i, j] = -1;
        2. shortest_distance[i, j] == np.inf, shortest_path[i, j] = -2;
        """
        self.raw_graph = raw_graph
        self.index2osm_id = index2osm_id
        self.shortest_distance = shortest_distance
        self.shortest_path = shortest_path
        self.access_index = access_index
        self.adjacent_location_osm_index = adjacent_location_osm_index
        self.adjacent_location_driven_distance = adjacent_location_driven_distance
        self.adjacent_location_goal_index = adjacent_location_goal_index
        self.index2location = index2location
        index_number = shortest_distance.shape[0]
        self.index_set = np.array(list(range(index_number)), dtype=np.int16)

    def _get_random_osm_index(self) -> int:
        """
        随机生成一个osm_index
        :return:
        """
        n, _ = self.shortest_distance.shape
        return np.random.choice(self.index_set)

    def get_shortest_distance_by_osm_index(self, osm_index1: int, osm_index2: int) -> float:
        return self.shortest_distance[osm_index1, osm_index2]

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

    def get_random_vehicle_locations(self, vehicle_number: int) -> List[VehicleLocation]:
        """
        用于返回一个随机车辆位置列表
        :return:
        """
        locations_index = np.random.choice(self.index_set, vehicle_number)
        locations = [VehicleLocation(locations_index[i]) for i in range(vehicle_number)]
        return locations

    def move_to_target_index(self, vehicle_location: VehicleLocation, target_index: int, could_drive_distance: float) -> NoReturn:
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
        else:
            vehicle_to_target_distance = FLOAT_ZERO
            while could_drive_distance > DISTANCE_EPS and not equal(could_drive_distance, DISTANCE_EPS):  # 还可以继续行驶的情况
                goal_index = self.shortest_path[vehicle_location.osm_index, target_index]
                vehicle_to_goal_distance = self.shortest_distance[vehicle_location.osm_index, goal_index]
                if vehicle_to_goal_distance - could_drive_distance < DISTANCE_EPS or \
                        equal(vehicle_to_goal_distance - could_drive_distance, DISTANCE_EPS):
                    could_drive_distance -= vehicle_to_goal_distance
                    vehicle_to_target_distance += vehicle_to_goal_distance
                else:
                    vehicle_location.set_location(vehicle_location.osm_index, could_drive_distance, goal_index)
                    vehicle_to_target_distance += could_drive_distance
                    break
                vehicle_location.osm_index = goal_index
        return vehicle_to_target_distance

    def random_move(self, vehicle_location: VehicleLocation, could_drive_distance: float) -> float:
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
                this_time_drive_distance += vehicle_to_goal_distance  # 增加车辆行驶的距离
                could_drive_distance -= vehicle_to_goal_distance  # 减小车辆可以行驶的距离
            else:  # 不可以行驶动goal_index上
                vehicle_location.driven_distance += could_drive_distance  # 这样车辆还是会在两节点之间但是需要修改已经行驶的距离
                this_time_drive_distance += could_drive_distance  # 得到车辆当前行驶的距离
                could_drive_distance = FLOAT_ZERO

            # 如果车辆还有可行驶的距离探索可达节点
            if could_drive_distance > DISTANCE_EPS and not equal(could_drive_distance, DISTANCE_EPS) and \
                    len(self.access_index[vehicle_location.osm_index]) != INT_ZERO:
                target_index = np.random.choice(self.access_index[vehicle_location.osm_index])
                this_time_drive_distance += self.move_to_target_index(vehicle_location, target_index, could_drive_distance)
        else:
            # 直接选择点随机行走
            # if len(self.access_index[vehicle_location.osm_index]) == INT_ZERO:  # 车当前的节点不可以到任何节点，那么就凭空移动，帮助其摆脱陷阱
            #     target_index = np.random.choice(range(len(self.shortest_distance)))
            #     vehicle_location.set_location(target_index)  # 凭空迁移
            # else:
            #     target_index = np.random.choice(self.access_index[vehicle_location.osm_index])
            #     this_time_drive_distance += self.move_to_target_index(vehicle_location, target_index, could_drive_distance)
            if len(self.adjacent_location_osm_index[vehicle_location.osm_index]) == INT_ZERO:  # 一分钟到不了任何节点，随机选择一个可以到达节点作为目标节点前进
                if len(self.access_index[vehicle_location.osm_index]) == INT_ZERO:  # 车当前的节点不可以到任何节点，那么就凭空移动，帮助其摆脱陷阱
                    target_index = np.random.choice(range(len(self.shortest_distance)))
                    vehicle_location.set_location(target_index)  # 凭空迁移
                else:
                    target_index = np.random.choice(self.access_index[vehicle_location.osm_index])
                    this_time_drive_distance += self.move_to_target_index(vehicle_location, target_index, could_drive_distance)
            else:
                idx = np.random.randint(0, len(self.adjacent_location_osm_index[vehicle_location.osm_index]))
                # idx = np.random.choice(range(len(self.adjacent_location_osm_index[vehicle_location.osm_index])))
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


class GridGraph(BaseGraph):
    """
    网格路网图
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

    def _convert_index_to_xy(self, osm_index: int):
        return osm_index // self.graph_size, osm_index % self.graph_size

    def _convert_xy_to_index(self, x: int, y: int):
        return x * self.graph_size + y

    def _get_shortest_distance_by_xy(self, x1: int, y1: int, x2: int, y2: int):
        return (np.abs(x1 - x2) + np.abs(y1 - y2)) * self.grid_size

    def _get_random_osm_index(self) -> int:
        """
        随机生成一个osm_index
        :return:
        """
        random_x = np.random.choice(self.x_list)
        random_y = np.random.choice(self.y_list)
        return self._convert_xy_to_index(random_x, random_y)

    def _get_best_next_xy(self, now_x: int, now_y: int, target_x: int, target_y: int):
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
        next_xy_lists = [(next_x, next_y, self._get_shortest_distance_by_xy(next_x, next_y, target_x, target_y))
                         for next_x, next_y in next_xy_lists]
        next_xy_lists.sort(key=lambda x: x[2])
        return next_xy_lists[0][0], next_xy_lists[0][1]

    def get_shortest_distance_by_osm_index(self, osm_index1: int, osm_index2: int):
        x1, y1 = self._convert_index_to_xy(osm_index1)
        x2, y2 = self._convert_index_to_xy(osm_index2)
        return self._get_shortest_distance_by_xy(x1, y1, x2, y2)

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
        distance_a = self.get_shortest_distance_by_osm_index(vehicle_location.osm_index, order_location.osm_index)  # 意义看文档
        distance_b = self.get_shortest_distance_by_osm_index(vehicle_location.goal_index, order_location.osm_index)  # 意义看文档
        distance_c = self.get_shortest_distance_by_osm_index(vehicle_location.osm_index, vehicle_location.goal_index)  # 意义看文档
        # 用于判断车是否从goal_index到order_location
        diff = (distance_c - vehicle_location.driven_distance + distance_b) - (distance_a + vehicle_location.driven_distance)
        return diff < FLOAT_ZERO or equal(diff, FLOAT_ZERO)

    def get_random_vehicle_locations(self, vehicle_number: int) -> List[VehicleLocation]:
        """
        用于返回一些随机车辆位置
        :return:
        """
        xs = np.random.choice(self.x_list, vehicle_number)
        ys = np.random.choice(self.y_list, vehicle_number)
        locations_index = [self._convert_xy_to_index(xs[i], ys[i]) for i in range(vehicle_number)]
        locations = [VehicleLocation(locations_index[i]) for i in range(vehicle_number)]
        return locations

    def move_to_target_index(self, vehicle_location: VehicleLocation, target_index: int, could_drive_distance: float) -> float:
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
        vehicle_to_target_distance = self.get_shortest_distance_by_osm_index(vehicle_location.osm_index, target_index)
        if vehicle_to_target_distance - could_drive_distance < DISTANCE_EPS or \
                equal(vehicle_to_target_distance - could_drive_distance, DISTANCE_EPS):  # 可以到target_index的情况
            vehicle_location.set_location(target_index)
        else:
            vehicle_to_target_distance = FLOAT_ZERO
            now_x, now_y = self._convert_index_to_xy(vehicle_location.osm_index)
            target_x, target_y = self._convert_index_to_xy(target_index)
            while could_drive_distance > DISTANCE_EPS and not equal(could_drive_distance, DISTANCE_EPS):  # 需要行驶的距离已经过小了的情况
                goal_x, goal_y = self._get_best_next_xy(now_x, now_y, target_x, target_y)
                vehicle_to_goal_distance = self.grid_size  # 因为只是移动一个格子, 所以只是移动一个小格子
                if vehicle_to_goal_distance - could_drive_distance < DISTANCE_EPS or \
                        equal(vehicle_to_goal_distance - could_drive_distance, DISTANCE_EPS):
                    could_drive_distance -= vehicle_to_goal_distance
                    vehicle_to_target_distance += vehicle_to_goal_distance
                else:
                    now_index = self._convert_xy_to_index(now_x, now_y)
                    goal_index = self._convert_xy_to_index(goal_x, goal_y)
                    vehicle_location.set_location(now_index, could_drive_distance, goal_index)
                    vehicle_to_target_distance += could_drive_distance
                    break
                now_x, now_y = goal_x, goal_y
            else:
                now_index = self._convert_xy_to_index(now_x, now_y)  # 正常退出情况需要更新此时车辆的位置
                vehicle_location.set_location(now_index)
        return vehicle_to_target_distance

    def random_move(self, vehicle_location: VehicleLocation, could_drive_distance: float) -> float:
        """
        :param vehicle_location:  车辆当前的位置
        :param could_drive_distance: 车辆可行行驶的距离
        ------
        注意： 这个函数会修改vehicle_location的值!!!!!!
        在一个时间间隔内，车辆随机路网上行驶
        """
        time_slot_drive_distance = FLOAT_ZERO
        if vehicle_location.is_between:  # 车辆处于两个节点之间
            vehicle_to_goal_distance = self.get_shortest_distance_by_osm_index(vehicle_location.osm_index, vehicle_location.goal_index) - \
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
            target_index = self._get_random_osm_index()
            if target_index != vehicle_location.osm_index:
                time_slot_drive_distance += self.move_to_target_index(vehicle_location, target_index, could_drive_distance)

        return time_slot_drive_distance

