#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/27
from network.transport_network.location import OrderLocation
from network.transport_network.graph import NetworkGraph
from typing import List
from setting import AVERAGE_SPEED

__all__ = ["SingleOrderBid", "BiddingStrategy", "AdditionalValueStrategy", "PickUpDistanceStrategy"]


class SingleOrderBid:
    __slots__ = ["route_plan", "value"]

    def __init__(self, route_plan: List[OrderLocation], value):
        self.route_plan = route_plan
        self.value = value


class BiddingStrategy:
    __slots__ = []

    def get_bids(self, vehicle, orders, network: NetworkGraph, current_time: int):
        raise NotImplementedError


class AdditionalValueStrategy(BiddingStrategy):  # 增加量的投标策略 例如增加成本 增加利润
    __slots__ = ["optimizer", "route_planning_method"]

    def __init__(self, optimizer, route_planning_method):
        self.optimizer = optimizer
        self.route_planning_method = route_planning_method

    def get_bids(self, vehicle, orders, network: NetworkGraph, current_time: int):
        compute_value_method = self.optimizer.compute_value_method
        original_value = compute_value_method(vehicle, vehicle.route_plan, current_time, network)
        vehicle_bids = {}
        for order in orders:
            new_value = self.route_planning_method(vehicle, order, network, current_time, self.optimizer)
            route_plan = self.optimizer.best_route_plan
            vehicle_bids[order] = SingleOrderBid(route_plan, new_value - original_value)
        return vehicle_bids


class PickUpDistanceStrategy(BiddingStrategy):  # 采用接送距离作为投标距离

    def get_bids(self, vehicle, orders, network: NetworkGraph, current_time: int):
        from agent.route_planning import compute_cost
        original_cost = compute_cost(vehicle, vehicle.route_plan, current_time, network)
        vehicle_bids = {}
        for order in orders:
            rest_of_distance = network.compute_vehicle_rest_pick_up_distance(vehicle.location, order.pick_location)
            rest_of_time = order.request_time + order.max_wait_time - current_time
            if rest_of_distance < AVERAGE_SPEED * rest_of_time and \
                    0 < original_cost < order.trip_fare and \
                    order.n_riders < vehicle.n_seats:
                vehicle_bids[order] = SingleOrderBid(None, rest_of_distance)  # TODO后续需要修改统一
        return vehicle_bids
