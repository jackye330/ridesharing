#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/3
"""
用于统一接口
"""
from typing import List
from constant import FIRST_INDEX
from constant import FLOAT_ZERO
from constant import INT_ZERO
from env.graph import BaseGraph
from env.location import GeoLocation
from env.location import OrderLocation
from env.location import VehicleLocation
from agent.route import Route
from setting import DISTANCE_EPS
from utility import is_enough_small

__all__ = ["Network", "network"]


class Network:
    __slots__ = ["_graph"]

    def __init__(self, graph: BaseGraph):
        self._graph = graph

    def get_shortest_distance(self, location1: GeoLocation, location2: GeoLocation) -> float:
        return self._graph.get_shortest_distance_by_osm_index(location1.osm_index, location2.osm_index)

    def generate_random_vehicle_locations(self, vehicle_number: int) -> List[VehicleLocation]:
        return self._graph.generate_random_vehicle_locations(vehicle_number)

    def can_move_to_goal_index(self, vehicle_location: VehicleLocation, order_location: OrderLocation) -> bool:
        return self._graph.can_move_to_goal_index(vehicle_location, order_location)

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
        graph = self._graph
        distance_a = graph.get_shortest_distance_by_osm_index(vehicle_location.osm_index, order_location.osm_index)  # 意义看文档
        if vehicle_location.is_between:
            if self.can_move_to_goal_index(vehicle_location, order_location):  # 车无法通过goal_index到达order_location
                distance_b = graph.get_shortest_distance_by_osm_index(vehicle_location.goal_index, order_location.osm_index)  # 意义看文档
                distance_c = graph.get_shortest_distance_by_osm_index(vehicle_location.osm_index, vehicle_location.goal_index)  # 意义看文档
                rest_pick_up_distance = distance_c - vehicle_location.driven_distance + distance_b
            else:
                rest_pick_up_distance = distance_a + vehicle_location.driven_distance
        else:
            rest_pick_up_distance = distance_a

        return rest_pick_up_distance

    def simulate_drive_on_route(self, vehicle_location: VehicleLocation, route_list: List[OrderLocation]):
        """
        一个模拟器，模拟车辆按照一个行驶路线行走，注意这个只是一个模拟器，不会修改车辆任何值，每一次yield 出当前行驶到订单位置，和行驶的距离
        :param vehicle_location: 车辆坐标
        :param route_list: 行驶路线
        ------
        注意：
        这个函数不可以修改 vehicle_location的值，并且这个函数是一个生成器
        """
        if len(route_list) == INT_ZERO:
            raise Exception("无法处理没有订单的路线规划")
        graph = self._graph
        move_osm_index = vehicle_location.osm_index
        if vehicle_location.is_between:  # 如果车两个节点之间需要提前行走一段距离
            if self.can_move_to_goal_index(vehicle_location, route_list[FIRST_INDEX]):  # 车无法通过goal_index到达order_location
                move_osm_index = vehicle_location.goal_index
                osm_to_goal_distance = graph.get_shortest_distance_by_osm_index(vehicle_location.osm_index, vehicle_location.goal_index)
                pre_drive_distance = osm_to_goal_distance - vehicle_location.driven_distance
            else:
                pre_drive_distance = vehicle_location.driven_distance
        else:
            pre_drive_distance = FLOAT_ZERO

        for i in range(len(route_list)):
            order_location = route_list[i]
            vehicle_to_order_distance = graph.get_shortest_distance_by_osm_index(move_osm_index, order_location.osm_index)
            if i == FIRST_INDEX:  # vehicle_to_order_distance表示每一次车辆到一个订单节点的距离，首先车辆到订单的距离加上这个预先行驶距离
                vehicle_to_order_distance += pre_drive_distance
            yield order_location, vehicle_to_order_distance
            move_osm_index = order_location.osm_index

    def drive_on_random(self, vehicle_location: VehicleLocation, could_drive_distance: float) -> float:
        return self._graph.move_to_random_index(vehicle_location, could_drive_distance)

    def drive_on_route(self, vehicle_location: VehicleLocation, route: Route, could_drive_distance: float):
        """
        在一个时间间隔内，车辆按照自己的路线进行行驶
        :param vehicle_location: 车俩当前的位置
        :param route: 车辆当前的行驶路线
        :param could_drive_distance: 车辆可行行驶的距离
        ------
        注意：
        这个函数会修改vehicle_location的值
        """
        graph = self._graph
        if vehicle_location.is_between:  # 当前车辆在两点之间
            if self.can_move_to_goal_index(vehicle_location, route[FIRST_INDEX]):  # 当前车辆需要向location.goal_index行驶
                vehicle_to_goal_distance = graph.get_shortest_distance_by_osm_index(vehicle_location.osm_index, vehicle_location.goal_index) - \
                                           vehicle_location.driven_distance
                # 判断是否可以到location.goal_index
                # 1. vehicle 到 goal_index 的距离远远小于could_drive_distance
                # 2. vehicle 到 goal_index 的距离只比could_drive_distance大DISTANCE_EPS
                if is_enough_small(vehicle_to_goal_distance - could_drive_distance, DISTANCE_EPS):  # 是否可以继续行驶
                    pre_drive_distance = vehicle_to_goal_distance
                    vehicle_location.set_location(vehicle_location.goal_index)  # 由于不在两点之间了需要重置goal_index和相应的一些设置
                else:
                    pre_drive_distance = could_drive_distance
                    vehicle_location.driven_distance += could_drive_distance  # 还在之前的两个节点之间只需要修改行驶距离就可以，只不过是向goal_index方向行驶
            else:
                # 判断是否可以回到上一个出发节点
                # 1. vehicle 到 vehicle_location 的距离远远小于should_drive_distance
                # 2. vehicle 到 vehicle_location 的距离只是比could_drive_distance大DISTANCE_EPS
                driven_distance = vehicle_location.driven_distance
                if is_enough_small(driven_distance - could_drive_distance, DISTANCE_EPS):  # 是否可以继续行驶
                    pre_drive_distance = driven_distance
                    vehicle_location.reset()  # 又回到osm_index上
                else:
                    pre_drive_distance = could_drive_distance
                    vehicle_location.driven_distance -= could_drive_distance  # 还在之前的两个节点之间只需要修改行驶距离就可以，只不过是向osm_index
        else:
            pre_drive_distance = FLOAT_ZERO

        could_drive_distance -= pre_drive_distance  # 减少预先行驶的距离

        if not vehicle_location.is_between or not is_enough_small(could_drive_distance, DISTANCE_EPS):
            # 如果车辆不在两点之间而就在一个点上 或者 车辆的可行使距离还有很多的情况下
            # 第一个订单节点的位置有两种可能性第一种订单节点在车辆处于两个节点中任何一个或者是在别的位置
            for covered_index, order_location in enumerate(route):  # 现在开始探索路线规划的各个节点
                vehicle_to_order_distance = graph.get_shortest_distance_by_osm_index(vehicle_location.osm_index, order_location.osm_index)

                if is_enough_small(vehicle_to_order_distance - could_drive_distance, DISTANCE_EPS):  # 当此订单节点是可以到达的情况
                    could_drive_distance -= vehicle_to_order_distance  # 更新当前车辆需要行驶的距离
                    vehicle_location.osm_index = order_location.osm_index  # 更新车辆坐标

                    if covered_index == FIRST_INDEX:
                        vehicle_to_order_distance += pre_drive_distance
                    yield True, covered_index, order_location, vehicle_to_order_distance

                    if is_enough_small(could_drive_distance, DISTANCE_EPS):  # 需要行驶的距离已经过小了的情况下直接拉到对应的位置
                        break
                else:  # 订单节点路长过大无法到达的情况, 需要进行精细调整
                    target_index = order_location.osm_index
                    vehicle_to_target_distance = graph.move_to_target_index(vehicle_location, target_index, could_drive_distance)
                    if covered_index == FIRST_INDEX:  # 如果是第一个订单就是不可达的那么要考虑之前行驶的距离
                        vehicle_to_target_distance += pre_drive_distance
                    yield False, covered_index - 1, order_location, vehicle_to_target_distance
                    break
        else:  # 车辆一开始就把所有可以运行的距离都运行，车辆根本就不可能到任何一个订单节点
            yield False, -1, None, pre_drive_distance


def _generate_network():
    from setting import REAL
    from setting import GRID
    from setting import EXPERIMENTAL_MODE
    from env.graph import RealGraph
    from env.graph import GridGraph
    import numpy as np

    if EXPERIMENTAL_MODE == REAL:
        import os
        import pickle
        # import osmnx as ox
        from setting import GEO_DATA_FILE
        geo_data_base_folder = GEO_DATA_FILE["base_folder"]
        graph_file = GEO_DATA_FILE["graph_file"]
        # osm_id2index_file = os.path.join(geo_data_base_folder, GEO_DATA_FILE["osm_id2index_file"])
        # index2osm_id_file = os.path.join(geo_data_base_folder, GEO_DATA_FILE["index2osm_id_file"])
        shortest_distance_file = os.path.join(geo_data_base_folder, GEO_DATA_FILE["shortest_distance_file"])
        shortest_path_file = os.path.join(geo_data_base_folder, GEO_DATA_FILE["shortest_path_file"])
        access_index_file = os.path.join(geo_data_base_folder, GEO_DATA_FILE["access_index_file"])
        adjacent_location_osm_index_file = os.path.join(geo_data_base_folder, GEO_DATA_FILE["adjacent_location_osm_index_file"])
        adjacent_location_driven_distance_file = os.path.join(geo_data_base_folder, GEO_DATA_FILE["adjacent_location_driven_distance_file"])
        adjacent_location_goal_index_file = os.path.join(geo_data_base_folder, GEO_DATA_FILE["adjacent_location_goal_index_file"])

        # raw_graph = ox.load_graphml(graph_file, folder=geo_data_base_folder)
        shortest_distance = np.load(shortest_distance_file)
        shortest_path = np.load(shortest_path_file)
        with open(access_index_file, "rb") as file:
            access_index = pickle.load(file)
        with open(adjacent_location_osm_index_file, "rb") as file:
            adjacent_location_osm_index = pickle.load(file)
        with open(adjacent_location_driven_distance_file, "rb") as file:
            adjacent_location_driven_distance = pickle.load(file)
        with open(adjacent_location_goal_index_file, "rb") as file:
            adjacent_location_goal_index = pickle.load(file)
        # with open(osm_id2index_file, "rb") as file:
        #     osm_id2index = pickle.load(file)
        # with open(index2osm_id_file, "rb") as file:
        #     index2osm_id = pickle.load(file)

        # index2location = {osm_id2index[node[0]]: (node[1]['x'], node[1]['y']) for node in raw_graph.nodes(data=True)}

        graph = RealGraph(
            shortest_distance=shortest_distance,
            shortest_path=shortest_path,
            access_index=access_index,
            adjacent_location_osm_index=adjacent_location_osm_index,
            adjacent_location_driven_distance=adjacent_location_driven_distance,
            adjacent_location_goal_index=adjacent_location_goal_index)
    elif EXPERIMENTAL_MODE == GRID:
        from setting import GRAPH_SIZE
        from setting import GRID_SIZE
        graph = GridGraph(graph_size=GRAPH_SIZE, grid_size=GRID_SIZE)
    else:
        graph = None

    return Network(graph)


network = _generate_network()  # 单例
