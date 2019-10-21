#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/3/6
import pickle
from typing import List, Tuple, Dict, Set

import numpy as np
import osmnx as ox
import pandas as pd
from networkx.classes.multidigraph import MultiDiGraph

from agent import GeoLocation
from agent import Order
from agent import OrderLocation
from agent import Vehicle


def initialize_environment(min_request_time: int, max_request_time: int, max_wait_times: List[int],
                           detour_ratios: List[float], vehicle_number: int, vehicle_speed: float) \
        -> Tuple[MultiDiGraph, np.ndarray, np.ndarray, np.ndarray, Dict[int, Set[Order]], List[Vehicle]]:
    """
    初始化环境
    :param max_wait_times: 订单最大等待时间集合
    :param detour_ratios: 订单最大绕路比集合
    :param min_request_time: 提取订单的最小请求时间
    :param max_request_time: 提取订单的最大请求时间
    :param vehicle_number: 生成的车辆数目
    :param vehicle_speed: 车辆的速度
    :return graph: 路网图
    :return shortest_distance：最短路径长度矩阵
    :return shortest_path: 最短路径矩阵
    :return shortest_path_with_minute: 1分钟以内的最短路径矩阵
    :return orders: 各个时刻的订单信息
    :return vehicles: 初始化的车辆列表
    """
    print("read data from disc")
    trip_order_data = pd.read_csv("./order_data/trip_order_data.csv")
    car_fuel_consumption_info = pd.read_csv("car_fuel_consumption_data/car_fuel_consumption_info.csv")
    graph = ox.load_graphml("Manhattan.graphml", folder="./network_data/")
    shortest_distance = np.load("./network_data/shortest_distance.npy")
    shortest_path = np.load("./network_data/shortest_path.npy")
    shortest_path_with_minute = np.load("./network_data/shortest_path_with_minute.npy")

    print("build osm_id to index map")
    osm_id2index = {osm_id: index for index, osm_id in enumerate(graph.nodes)}
    location_map = {osm_id2index[node[0]]: (node[1]['x'], node[1]['y']) for node in graph.nodes(data=True)}
    index2osm_id = {index: osm_id for osm_id, index in osm_id2index.items()}
    GeoLocation.set_location_map(location_map)
    GeoLocation.set_index2osm_id(index2osm_id)

    print("generate order data")
    trip_order_data = trip_order_data[min_request_time <= trip_order_data["time"]]
    trip_order_data = trip_order_data[max_request_time > trip_order_data["time"]]

    order_number = trip_order_data.shape[0]
    pick_up_index_series = trip_order_data["pick_up_index"].values
    drop_off_index_series = trip_order_data["drop_off_index"].values
    request_time_series = trip_order_data["time"].values
    receive_fare_series = (trip_order_data["total_amount"] - trip_order_data["tip_amount"]).values  # 我们不考虑订单的中tip成分
    n_riders_series = trip_order_data["passenger_count"].values

    orders = {}
    for request_time in range(min_request_time, max_request_time):
        orders[request_time] = set()

    for i in range(order_number):
        order_id = i
        start_location = OrderLocation(int(pick_up_index_series[i]), OrderLocation.PICK_UP_TYPE)
        end_location = OrderLocation(int(drop_off_index_series[i]), OrderLocation.DROP_OFF_TYPE)
        request_time = int(request_time_series[i])
        max_wait_time = np.random.choice(max_wait_times)
        order_distance = shortest_distance[start_location.osm_index, end_location.osm_index]
        if order_distance == 0.0:
            continue
        receive_fare = receive_fare_series[i]
        detour_ratio = np.random.choice(detour_ratios)
        n_riders = int(n_riders_series[i])
        order = Order(order_id, start_location, end_location, request_time, max_wait_time,
                      order_distance, receive_fare, detour_ratio, n_riders)
        orders[request_time].add(order)

    print("generate vehicle data")
    Vehicle.set_average_speed(vehicle_speed)
    car_osm_ids = np.random.choice(graph.nodes, size=vehicle_number)
    cars_info = car_fuel_consumption_info.sample(n=vehicle_number)
    vehicles = []
    for i in range(vehicle_number):
        vehicle_id = i
        location = GeoLocation(osm_id2index[int(car_osm_ids[i])])
        car_info = cars_info.iloc[i, :]
        available_seats = int(car_info["seats"])
        cost_per_distance = float(car_info["fuel_consumption"]) / 6.8 * 2.5 / 1.609344
        vehicle = Vehicle(vehicle_id, location, available_seats, cost_per_distance, Vehicle.WITHOUT_MISSION_STATUS)
        vehicles.append(vehicle)

    print("finish generate data")
    return graph, shortest_distance, shortest_path, shortest_path_with_minute, orders, vehicles


def plot_vehicle_in_graph(graph: MultiDiGraph, vehicles: List[Vehicle]):
    """
    在路网图上绘制车辆
    :param graph:
    :param vehicles:
    :return:
    """
    vehicle_nodes = set()
    for vehicle in vehicles:
        vehicle_nodes.add(GeoLocation.INDEX2OSM_ID[vehicle.location.osm_index])
    nc = ['r' if node in vehicle_nodes else 'none' for node in graph.nodes()]
    ox.plot_graph(graph, node_color=nc)


def print_vehicles_info(vehicles: List[Vehicle]):
    for vehicle in vehicles:
        if vehicle.status == Vehicle.HAVE_MISSION_STATUS:
            print(vehicle)


def vo_bids2ov_match(candidate_orders: Set[Order], candidate_vehicles: Set[Vehicle],
                     bids: Dict[Vehicle, Dict[Order, float]])\
     -> Dict[Order, Dict[Vehicle, float]]:
    ov_match = {}
    for order in candidate_orders:
        ov_match[order] = {}
    for vehicle in candidate_vehicles:
        for order in candidate_orders:
            ov_match[order][vehicle] = bids[vehicle][order]
    return ov_match


def force_type_check(func):
    from inspect import signature
    sig = signature(func)

    def decorate_func(*args, **kwargs):

        bound_arguments = sig.bind(*args, **kwargs)

        for param, value in bound_arguments.arguments.items():
            if not isinstance(value, func.__annotations__[param]):
                raise Exception("{0} should be {1}, but you give {2}".
                                format(param, func.__annotations__[param], type(value)))
        for param, value in bound_arguments.kwargs.items():
            if not isinstance(value, func.__annotations__[param]):
                raise Exception("{0} should be {1}, but you give {2}".
                                format(param, func.__annotations__[param], type(value)))

        return func(*args, **kwargs)

    return decorate_func


def print_execute_time(func):
    import time

    def decorate_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print("{0} has elapsed {1} s".format(func.__name__, time.time() - start_time))
        return result

    return decorate_func
