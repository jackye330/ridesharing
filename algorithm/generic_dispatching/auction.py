#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/26
from typing import List, Tuple, Set, Dict
import numpy as np

from agent.order import Order
from agent.vehicle import Vehicle


def orders_dispatch_with_sequence_auction(
        shortest_distance: np.ndarray, shortest_path: np.ndarray, shortest_path_with_minute: np.ndarray,
        adjacent_nodes: np.ndarray, orders: Set[Order], vehicles: List[Vehicle], current_time: int) \
        -> Tuple[Set[Order], Dict[Vehicle, List[Tuple[Order, float]]], float, float, float, float, float, float]:
    """
    :param shortest_distance:
    :param shortest_path:
    :param shortest_path_with_minute:
    :param adjacent_nodes:
    :param orders:
    :param vehicles:
    :param current_time:
    :return:
    """
    # 对于每一个订单取距离最近的车分配，如果这个车的调度是可以满足条件的，而且容量是可以满足需求的
    dispatched_orders = set()
    payments = {}
    social_welfare = 0.0
    social_cost = 0.0
    total_payment = 0.0
    total_utility = 0.0
    total_profit = 0.0

    # 对于订单按照价值排序(按sw排序)
    candidate_orders = list(sorted(orders, key=lambda _order: _order.trip_fare))

    for order in candidate_orders:

        candidate_vehicles_bids = list()

        for vehicle in vehicles:
            # 判断投标可行性 人数要满足条件 还有距离要可达
            if order.n_riders > vehicle.n_seats:
                continue
            two_location_distance = shortest_distance[vehicle.location.osm_index, order.start_location.osm_index]
            rest_of_time = order.max_wait_time + order.request_time - current_time
            # 车处于两个节点中间
            if vehicle.is_between == Vehicle.IS_BETWEEN_TWO_INDEX:
                # 顺路或者不能反向行驶
                if shortest_distance[vehicle.goal_index, order.start_location.osm_index] + \
                        shortest_distance[vehicle.location.osm_index, vehicle.goal_index] - vehicle.between_distance < \
                        two_location_distance + vehicle.between_distance or \
                        shortest_distance[vehicle.goal_index, vehicle.location.osm_index] == np.inf:
                            if shortest_distance[vehicle.goal_index, order.start_location.osm_index] + \
                                    shortest_distance[
                                        vehicle.location.osm_index, vehicle.goal_index] - vehicle.between_distance > \
                                    rest_of_time * Vehicle.AVERAGE_SPEED:
                                    continue
                else:
                    if two_location_distance + vehicle.between_distance > rest_of_time * Vehicle.AVERAGE_SPEED:
                        continue
            else:
                if two_location_distance > rest_of_time * Vehicle.AVERAGE_SPEED:
                    continue

            original_cost = vehicle.compute_cost(shortest_distance, vehicle.route_plan, current_time)  # 计算没有投标的时候的费用
            current_cost, _ = vehicle.find_route_plan(shortest_distance, order, current_time)  # 计算已经添加这个订单之后的费用
            additional_cost = current_cost - original_cost  # 计算费用的增加量

            if 0 <= additional_cost <= order.trip_fare:
                candidate_vehicles_bids.append((vehicle, additional_cost))

        if len(candidate_vehicles_bids) == 0:
            pass
        else:
            dispatched_orders.add(order)
            candidate_vehicles_bids.sort(key=lambda bid: bid[1])
            winner_vehicle = candidate_vehicles_bids[0][0]
            winner_vehicle.n_seats -= order.n_riders
            _, route_plan = winner_vehicle.find_route_plan(shortest_distance, order, current_time)
            winner_vehicle.route_plan = route_plan
            order.belong_to_vehicle = winner_vehicle

            if len(candidate_vehicles_bids) == 1:
                payment = candidate_vehicles_bids[0][1]
            else:
                payment = candidate_vehicles_bids[1][1]  # 次价支付

            if winner_vehicle not in payments:
                payments[winner_vehicle] = [(order, payment)]
            else:
                payments[winner_vehicle].append((order, payment))

            social_welfare += (order.trip_fare - candidate_vehicles_bids[0][1])  # order.trip_fare - additional_cost
            social_cost += candidate_vehicles_bids[0][1]
            total_payment += payment
            total_utility += (payment - candidate_vehicles_bids[0][1])
            total_profit += (order.trip_fare - payment)

    vehicles.sort(key=lambda v: v.vehicle_id)
    form_location = {}
    for vehicle in vehicles:
        form_location[vehicle] = vehicle.location.osm_index
    # 车辆更新位置
    for vehicle in vehicles:
        # # 未激活车辆位置不变
        # if vehicle.activated == Vehicle.NEGATIVE:
        #     continue
        if vehicle not in payments:
            # 没有在胜者集合的而且没有路线目标的随机更新下一个位置
            if vehicle.status == Vehicle.WITHOUT_MISSION_STATUS:
                vehicle.update_random_location(shortest_distance, shortest_path_with_minute, adjacent_nodes)
            else:
                vehicle.update_order_location(shortest_distance, shortest_path)
            continue
        # 如果是胜者者的话就要决定最优路线而且还要更新订单的状态
        vehicle.update_order_location(shortest_distance, shortest_path)
    for vehicle in vehicles:
        if form_location[vehicle] == vehicle.location.osm_index:
            print("位置没变", vehicle.vehicle_id, vehicle.location.osm_index, vehicle.goal_index, vehicle.route_plan)
    empty_vehicle_num = 0
    for vehicle in vehicles:
        # 统计空闲车的数量
        if vehicle.status == vehicle.WITHOUT_MISSION_STATUS:
            empty_vehicle_num += 1
    empty_vehicle_rate = empty_vehicle_num / len(vehicles)
    return dispatched_orders, payments, social_welfare, social_cost, total_payment, total_utility, total_profit, empty_vehicle_rate


if __name__ == '__main__':
    # TODO：测试algorithm的功能
    pass

