#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/2/22
from typing import List, NoReturn, Set

from agent.platform import Platform
from agent.proxy_bidder import AdditionalProfitBidder, AdditionalCostBidder
from agent.vehicle import Vehicle, generate_real_vehicles, generate_grid_vehicles
from algorithm.generic_dispatching.auction import second_price_sequence_auction
from algorithm.generic_dispatching.baseline import sparp_mechanism, nearest_vehicle_dispatching
from algorithm.route_planning.planner import InsertingPlanner, ReschedulingPlanner
from algorithm.simple_dispatching.auction import vcg_mechanism, greedy_mechanism
from setting import INSERTING, RESCHEDULING, VEHICLE_SPEED
from setting import MINIMIZE_COST, MAXIMIZE_PROFIT
from setting import REAL_TRANSPORT, GRID_TRANSPORT, ADDITIONAL_COST_STRATEGY, ADDITIONAL_PROFIT_STRATEGY, NEAREST_DISPATCHING, VCG_MECHANISM, GM_MECHANISM, SPARP_MECHANISM, SEQUENCE_AUCTION
from env.graph import BaseGraph, generate_grid_graph, generate_real_graph
from env.order import Order, real_order_generator, grid_order_generator
from env.network import Network
from setting import BIDDING_STRATEGY
from setting import DISPATCHING_METHOD
from setting import EXPERIMENTAL_MODE
from setting import TIME_SLOT, VEHICLE_NUMBER, MIN_REQUEST_TIME, MIN_REQUEST_DAY, SECOND_OF_DAY, INT_ZERO


class Simulator:
    __slots__ = [
        "network", "vehicles", "platform", "time_slot", "current_time",
        "social_welfare_trend", "social_cost_trend", "total_driver_rewards_trend", "total_driver_payoffs_trend", "platform_profit_trend",
        "accumulate_service_ratio_trend", "total_orders_number_trend", "serviced_orders_number_trend",
        "empty_vehicle_number_trend", "total_vehicle_number_trend", "empty_vehicle_ratio_trend",
        "accumulate_service_distance_trend", "accumulate_random_distance_trend",
        "each_orders_service_time_trend", "each_orders_wait_time_trend", "each_orders_detour_ratio_trend",
        "each_vehicles_reward", "each_vehicles_profit", "each_vehicles_finish_order_number",
        "each_vehicles_service_distance", "each_vehicles_random_distance",
        "bidding_time_trend", "running_time_trend",
    ]

    def __init__(self):
        # 模拟器构造环境
        if BIDDING_STRATEGY == ADDITIONAL_COST_STRATEGY:
            proxy_bidder = AdditionalCostBidder()
        elif BIDDING_STRATEGY == ADDITIONAL_PROFIT_STRATEGY:
            proxy_bidder = AdditionalProfitBidder()
        else:
            raise Exception("目前还没有实现其他投标方式")

        from setting import ROUTE_PLANNING_GOAL
        from setting import ROUTE_PLANNING_METHOD

        if ROUTE_PLANNING_GOAL == MINIMIZE_COST:
            from algorithm.route_planning.optimizer import MinimizeCostOptimizer
            optimizer = MinimizeCostOptimizer()
        elif ROUTE_PLANNING_GOAL == MAXIMIZE_PROFIT:
            from algorithm.route_planning.optimizer import MaximizeProfitOptimizer
            optimizer = MaximizeProfitOptimizer()
        else:
            optimizer = None

        if ROUTE_PLANNING_METHOD == INSERTING:
            route_planner = InsertingPlanner(optimizer)
        elif ROUTE_PLANNING_METHOD == RESCHEDULING:
            route_planner = ReschedulingPlanner(optimizer)
        else:
            raise Exception("目前还没有实现其他的路线规划方式")

        if EXPERIMENTAL_MODE == REAL_TRANSPORT:
            BaseGraph.set_generate_graph_function(generate_real_graph)
            Vehicle.set_generate_vehicles_function(generate_real_vehicles)
            Order.set_order_generator(real_order_generator)
        elif EXPERIMENTAL_MODE == GRID_TRANSPORT:
            BaseGraph.set_generate_graph_function(generate_grid_graph)
            Vehicle.set_generate_vehicles_function(generate_grid_vehicles)
            Order.set_order_generator(grid_order_generator)
        else:
            raise Exception("目前还没有实现其实验模式")

        if DISPATCHING_METHOD == VCG_MECHANISM:
            mechanism = vcg_mechanism
        elif DISPATCHING_METHOD == GM_MECHANISM:
            mechanism = greedy_mechanism
        elif DISPATCHING_METHOD == SPARP_MECHANISM:
            mechanism = sparp_mechanism
        elif DISPATCHING_METHOD == SEQUENCE_AUCTION:
            mechanism = second_price_sequence_auction
        elif DISPATCHING_METHOD == NEAREST_DISPATCHING:
            mechanism = nearest_vehicle_dispatching
        else:
            raise Exception("目前还没有实现其他类型的订单分配机制")

        network = Network(BaseGraph.generate_graph())
        platform = Platform(mechanism)
        vehicles = Vehicle.generate_vehicles(
            vehicle_number=VEHICLE_NUMBER,
            vehicle_speed=VEHICLE_SPEED,
            time_slot=TIME_SLOT,
            proxy_bidder=proxy_bidder,
            route_planner=route_planner,
            network=network)

        # 初始化模拟器的变量
        self.platform: Platform = platform
        self.vehicles: List[Vehicle] = vehicles
        self.network: Network = network
        self.time_slot: int = TIME_SLOT
        self.current_time = MIN_REQUEST_DAY * SECOND_OF_DAY + MIN_REQUEST_TIME
        self.social_welfare_trend = list()
        self.social_cost_trend = list()
        self.total_driver_rewards_trend = list()
        self.total_driver_payoffs_trend = list()
        self.platform_profit_trend = list()
        self.total_orders_number_trend = list()
        self.serviced_orders_number_trend = list()
        self.accumulate_service_ratio_trend = list()
        self.empty_vehicle_number_trend = list()
        self.total_vehicle_number_trend = list()
        self.empty_vehicle_ratio_trend = list()
        self.bidding_time_trend = list()
        self.running_time_trend = list()
        self.accumulate_service_distance_trend = list()
        self.accumulate_random_distance_trend = list()
        self.each_orders_service_time_trend = list()
        self.each_orders_wait_time_trend = list()
        self.each_orders_detour_ratio_trend = list()
        self.each_vehicles_reward = list()
        self.each_vehicles_profit = list()
        self.each_vehicles_finish_order_number = list()
        self.each_vehicles_service_distance = list()
        self.each_vehicles_random_distance = list()

    def trace_vehicles_info(self, print_vehicle=False) -> NoReturn:
        """
        更新车辆信息
        :return:
        """
        mechanism = self.platform.dispatching_mechanism
        empty_vehicle_number = INT_ZERO
        total_vehicle_number = INT_ZERO
        for vehicle in self.vehicles:
            if not vehicle.is_activated:
                continue
            total_vehicle_number += 1
            if vehicle not in mechanism.dispatched_vehicles and not vehicle.have_service_mission():
                vehicle.drive_on_random(self.network)
                empty_vehicle_number += 1
            else:
                if vehicle in mechanism.dispatched_vehicles:
                    dispatching_result = mechanism.dispatched_results[vehicle]
                    for order in dispatching_result.orders:
                        order.set_belong_vehicle(vehicle)
                    vehicle.set_route(dispatching_result.driver_route)
                    vehicle.increase_earn_reward(dispatching_result.driver_reward)
                    vehicle.increase_earn_profit(dispatching_result.driver_profit)

                finish_orders = vehicle.drive_on_route(self.current_time, self.network)  # 下一个 time_slot 车的位置
                for order in finish_orders:
                    self.each_orders_wait_time_trend.append(order.real_wait_time)
                    self.each_orders_service_time_trend.append(order.real_service_time)
                    self.each_orders_detour_ratio_trend.append(order.real_detour_ratio)

                if print_vehicle:
                    print(vehicle)
            # vehicle.leave_platform()  # TODO 日后去解决这个问题
        self.empty_vehicle_number_trend.append(empty_vehicle_number)
        self.total_vehicle_number_trend.append(total_vehicle_number)
        self.empty_vehicle_ratio_trend.append(empty_vehicle_number / total_vehicle_number)

    def finish_all_orders(self, vehicles: List[Vehicle]):
        for vehicle in vehicles:
            if not vehicle.is_activated:
                continue
            if vehicle.have_service_mission():
                finish_orders = vehicle.drive_on_route(self.current_time, self.network)
                for order in finish_orders:
                    self.each_orders_wait_time_trend.append(order.real_wait_time)
                    self.each_orders_service_time_trend.append(order.real_service_time)
                    self.each_orders_detour_ratio_trend.append(order.real_detour_ratio)
                print(vehicle.have_service_mission())

            self.each_vehicles_reward.append(vehicle.earn_reward)
            self.each_vehicles_profit.append(vehicle.earn_profit)
            self.each_vehicles_finish_order_number.append(vehicle.finish_orders_number)
            self.each_vehicles_service_distance.append(vehicle.service_driven_distance)
            self.each_vehicles_random_distance.append(vehicle.random_driven_distance)
            if vehicle.have_service_mission():
                raise Exception("有这么长订单路线吗")

    def summary_each_round_result(self, new_orders: Set[Order]) -> NoReturn:
        """
        总结这次分配的结果
        :param new_orders: 新的订单
        :return:
        """
        mechanism = self.platform.dispatching_mechanism
        self.social_welfare_trend.append(mechanism.social_welfare)
        self.social_cost_trend.append(mechanism.social_cost)
        self.total_driver_rewards_trend.append(mechanism.total_driver_rewards)
        self.total_driver_payoffs_trend.append(mechanism.total_driver_payoffs)
        self.platform_profit_trend.append(mechanism.platform_profit)
        self.serviced_orders_number_trend.append(len(mechanism.dispatched_orders))
        self.total_orders_number_trend.append(len(new_orders))
        self.accumulate_service_ratio_trend.append(sum(self.serviced_orders_number_trend) / sum(self.total_orders_number_trend))
        self.bidding_time_trend.append(mechanism.bidding_time)
        self.running_time_trend.append(mechanism.running_time)
        self.accumulate_service_distance_trend.append(sum([vehicle.service_driven_distance for vehicle in self.vehicles if vehicle.is_activated]))
        self.accumulate_random_distance_trend.append(sum([vehicle.random_driven_distance for vehicle in self.vehicles if vehicle.is_activated]))

    def simulate(self):
        for current_time, new_orders in Order.order_generator(self.time_slot, self.network):
            self.current_time = current_time
            self.platform.round_based_process(self.vehicles, new_orders, self.current_time, self.network)  # 订单分发和司机定价
            self.trace_vehicles_info()  # 车辆更新信息
            self.summary_each_round_result(new_orders)  # 统计匹配结果
            print(self.empty_vehicle_ratio_trend[-1], self.accumulate_service_ratio_trend[-1])

        # 等待所有车辆完成订单之后结束
        self.current_time += self.time_slot
        Vehicle.set_could_drive_distance(10000000.0)

    def load_env(self):
        """
        首先加载环境，然后
        :return:
        """
        pass

    def save_result(self, file_name):
        import pickle
        result = [
            self.social_welfare_trend,
            self.social_cost_trend,
            self.total_driver_rewards_trend,
            self.total_driver_payoffs_trend,
            self.platform_profit_trend,
            self.total_orders_number_trend,
            self.serviced_orders_number_trend,
            self.accumulate_service_ratio_trend,
            self.empty_vehicle_ratio_trend,
            self.bidding_time_trend,
            self.running_time_trend,
            self.accumulate_service_distance_trend,
            self.accumulate_random_distance_trend,
            self.each_orders_wait_time_trend,
            self.each_orders_service_time_trend,
            self.each_orders_detour_ratio_trend,
            self.each_vehicles_reward,
            self.each_vehicles_profit,
            self.each_vehicles_finish_order_number,
            self.each_vehicles_service_distance,
            self.each_vehicles_random_distance
        ]
        with open(file_name, "wb") as file:
            pickle.dump(result, file)


if __name__ == '__main__':
    simulator = Simulator()
    simulator.simulate()
    simulator.save_result("./result1.pkl")
