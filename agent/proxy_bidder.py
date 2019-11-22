#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/27
from env.location import OrderLocation
from env.network import Network
from typing import List
from agent.vehicle import Vehicle
from env.order import Order
from utility import equal
from constant import EMPTY_SEQUENCE

__all__ = ["OrderBid", "ProxyBidder", "generate_proxy_bidder"]


class OrderBid:
    __slots__ = ["route_plan", "value"]

    def __init__(self, value, route_plan: List[OrderLocation]):
        self.value = value
        self.route_plan = route_plan

    def __setattr__(self, key, value):
        if key in ["route", "value"]:
            raise Exception("常量不可以修改")


class ProxyBidder:
    __slots__ = []

    def get_bids(self, vehicle: Vehicle, orders: List[Order], current_time: int, network: Network):
        raise NotImplementedError


class AdditionalValueBidder(ProxyBidder):
    """
    利用优化量的增值作为投标的代理投标者
    增值可以是 车辆成本的增加量/平台利润的增加量
    """
    __slots__ = []

    def get_bids(self, vehicle: Vehicle, orders: List[Order], current_time: int, network: Network):
        route_planner = vehicle.route_planner
        optimizer = route_planner.optimizer
        compute_value_method = optimizer.compute_value_method
        old_value = compute_value_method(vehicle, vehicle.route, current_time, network)
        vehicle_bids = {}
        for order in orders:
            if order.n_riders > vehicle.available_seats:  # 首先看人数是否可以对应
                continue
            route_planner.planning(vehicle, order, current_time)  # 这里面会检查接送距离/绕路比
            new_value = optimizer.get_best_value()
            new_route = optimizer.get_best_route()
            additional_value = new_value - old_value
            vehicle_bids[order] = OrderBid(additional_value, new_route)
        return vehicle_bids


class PickUpDistanceBidder(ProxyBidder):
    """
    将车辆距离订单的距离作为投标的代理投标者
    """

    def get_bids(self, vehicle: Vehicle, orders: List[Order], current_time: int, network: Network):
        vehicle_bids = {}
        for order in orders:
            if order.n_riders > vehicle.available_seats:
                continue
            rest_of_time = order.request_time + order.wait_time - current_time
            rest_of_distance = network.compute_vehicle_rest_pick_up_distance(vehicle.location, order.pick_location)
            if rest_of_distance < Vehicle.average_speed * rest_of_time or equal(rest_of_distance, Vehicle.average_speed * rest_of_time):
                vehicle_bids[order] = OrderBid(rest_of_distance, EMPTY_SEQUENCE)
        return vehicle_bids


def generate_proxy_bidder():
    """
    用于生成投标测的函数
    :return:
    """
    from setting import BIDDING_STRATEGY
    if BIDDING_STRATEGY == "PICK_DISTANCE":
        proxy_bidder = PickUpDistanceBidder()
    elif BIDDING_STRATEGY == "ADDITIONAL":
        proxy_bidder = AdditionalValueBidder()
    else:
        proxy_bidder = None

    return proxy_bidder




