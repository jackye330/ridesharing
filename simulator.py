#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/2/22

from env.graph import graph
from env.network import Network
from env.order import Order
from setting import VEHICLE_SPEED
from agent.platform import platform
from agent.vehicle import Vehicle


if __name__ == '__main__':
    network = Network(graph)
    vehicles = Vehicle.generate_vehicles(network)
    Vehicle.set_average_speed(VEHICLE_SPEED)
    for current_time, orders in Order.real_order_generator(network):
        platform.round_based_process(vehicles, orders, current_time, network)
