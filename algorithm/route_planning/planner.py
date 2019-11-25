#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/7
from typing import List, Dict
from agent.utility import VehicleType
from algorithm.route_planning.optimizer import Optimizer
from algorithm.route_planning.utility import PlanningResult
from algorithm.route_planning.utility import generate_route_by_insert_pick_drop_location, pre_check_need_to_planning, get_route_info
from env.location import OrderLocation, PickLocation, VehicleLocation
from env.network import Network
from env.order import Order
from setting import FLOAT_ZERO, INT_ZERO, POS_INF, FIRST_INDEX
from algorithm.route_planning.utility import RouteInfo
from utility import is_enough_small

__all__ = ["RoutePlanner", "InsertingPlanner", "ReschedulingPlanner"]


class RoutePlanner:
    __slots__ = ["_optimizer"]

    def __init__(self, optimizer: Optimizer):
        """
        :param optimizer: 路规划优化器
        """
        self._optimizer = optimizer

    def reset(self):
        self._optimizer.reset()

    def planning(self, vehicle_info: VehicleType, old_route: List[OrderLocation], order: Order, current_time: int, network: Network) -> PlanningResult:
        """
        利用车辆类型和已有的行驶路线规划将新订单插入后的行驶路线
        :param vehicle_info: 车辆信息
        :param old_route: 行驶路线
        :param order: 订单
        :param current_time: 当前时间
        :param network: 路网
        ------
        注意：
        这里面所有操作都不可以改变自身原有的变量的值 ！！！！！
        """
        raise NotImplementedError

    def summary_planning_result(self) -> PlanningResult:
        optimal_route = self._optimizer.optimal_route
        optimal_route_info = self._optimizer.optimal_route_info
        return PlanningResult(optimal_route, optimal_route_info)


class InsertingPlanner(RoutePlanner):
    """
    这个路线规划框架是 微软亚洲研究院 在 ICDE2013 中的论文里面采用的方法
    论文名称是 T-Share: A Large-Scale Dynamic Taxi Ridesharing Service T-Share
    该方法是保证原有的订单起始点的相对顺序不变，然后将新订单的起始位置和结束位置插入到原先的行驶路线中。
    时间复杂度为 O(n^2*m) n为原先订单起始位置数目，m为进行优化的时间复度

    进阶版本： 我们还采用了An Efﬁcient Insertion Operator in Dynamic Ridesharing Services (ICDE2019) 里面的方法去剪枝 使得算法复杂度
    """
    __slots__ = []

    def __init__(self, optimizer: Optimizer):
        """
        :param optimizer: 路径优化器可以争对每一个输入的行驶路线优化内部的最优路径 选项MaximizeProfitOptimizer/MinimizeCostOptimizer
        """
        super().__init__(optimizer)

    @staticmethod
    def _generate_insert_pair(vehicle_type: VehicleType, route: List[OrderLocation], order: Order, current_time, network: Network):

        def _generate_pck_list() -> List[int]:
            """
            这个函数将返回一个向量，这个向量表示按照行驶路线，行驶到第i个位置的时候，车还剩多少座位
            我们规定0位置是车辆位置， 从1 到 n 位置表示行驶路线上的坐标，我们返回一个（n+1）长度的向量
            详细的定义请看 An Efﬁcient Insertion Operator in Dynamic Ridesharing Services (ICDE2019) 里面的解释，我们这里做了一定修改，这里面是剩余座位
            :return:
            """
            _pck_list = [vehicle_type.available_seats] * (n + 1)
            for _i in range(1, n + 1):
                if isinstance(route[_i - 1], PickLocation):
                    _pck_list[_i] = _pck_list[_i - 1] - route[_i - 1].belong_order.n_riders
                else:
                    _pck_list[_i] = _pck_list[_i - 1] + route[_i - 1].belong_order.n_riders
            return _pck_list

        def _generate_arr_dist() -> List[float]:
            """
            返回到第i个节点已经行驶了的距离
            :return:
            """
            _arr_list = [vehicle_type.service_driven_distance] * (n + 1)
            for _i in range(n):
                _arr_list[_i + 1] = _arr_list[_i] + (network.get_shortest_distance(v_loc, route[_i]) if _i == FIRST_INDEX else network.get_shortest_distance(route[_i-1], route[_i]))
            return _arr_list

        n = len(route)
        avg_speed: float = vehicle_type.vehicle_speed
        old_dists: float = vehicle_type.service_driven_distance
        v_loc: VehicleLocation = vehicle_type.location
        pck_list: List[int] = _generate_pck_list()
        arr_list: List[float] = _generate_arr_dist()

        for i in range(n + 1):
            if pck_list[i] < order.n_riders:  # 订单在这之后无法上车
                continue
            pre_dist = network.get_shortest_distance(v_loc, order.pick_location) if i == FIRST_INDEX else network.get_shortest_distance(route[i-1], order.pick_location)
            upper_time = (order.request_time + order.wait_time - current_time)
            if not network.is_smaller_bound_distance(arr_list[i] + pre_dist - old_dists, avg_speed * upper_time):  # 当前订单在物理距离上不可行
                continue
            for j in range(i, n + 1):
                if pck_list[j] < order.n_riders:  # 送到目的地的时间太靠后了，导致无法消除之前插入带来的影响
                    break
                yield i, j

    def planning(self, vehicle_info: VehicleType, old_route: List[OrderLocation], order: Order, current_time: int, network: Network) -> PlanningResult:
        self.reset()  # 优化器初始化重要！！！！
        if pre_check_need_to_planning(vehicle_info, order, current_time, network):  # 订单压根就是接不到的无需插入
            for i, j in self._generate_insert_pair(vehicle_info, old_route, order, current_time, network):
                new_route = generate_route_by_insert_pick_drop_location(old_route, order, i, j)
                new_route_info = get_route_info(vehicle_info, new_route, current_time, network)
                self._optimizer.optimize(new_route, new_route_info, vehicle_info.unit_cost)  # 优化器进行优化
        return super(InsertingPlanner, self).summary_planning_result()


class ReschedulingPlanner(RoutePlanner):
    """
    这个路线规划框架是 Mohammad Asghari 在 SIGSPATIAL-16 中论文里面采用的方法
    论文名称是 Price-aware Real-time Ride-sharing at Scale An Auction-based Approach
    该方法可以允许打乱原先的行驶路线的订单接送顺序，插入新订单的的起始位置。
    时间复杂度为 O(n!m) n为原先订单起始位置数目，m为进行优化的时间复度
    """
    __slots__ = []

    def __init__(self, optimizer: Optimizer):
        super().__init__(optimizer)

    def planning(self, vehicle_type: VehicleType, old_route: List[OrderLocation], order: Order, current_time: int, network: Network) -> PlanningResult:

        def _generate_remain_loc_list() -> List[OrderLocation]:
            """
            构造remain_list列表，只包含订单的起始点或者单独的订单终点
            """
            _order_set = set()
            _rem_loc_list = list()
            for order_location in old_route:  # 人数都不满足要求不用往后执行
                if isinstance(order_location, PickLocation):  # order_location 是一个订单的起始点直接加入
                    _rem_loc_list.append(order_location)
                    _order_set.add(order_location.belong_order)
                elif order_location.belong_order not in _order_set:  # 如果这一单是只有起始点的就直接加入
                    _rem_loc_list.append(order_location)
            _rem_loc_list.append(order.pick_location)
            return _rem_loc_list

        def _recursion(_cur_loc_list: List[OrderLocation], _rem_loc_list: List[OrderLocation], cur_seats: int, cur_dists: float):
            """
            递归函数，递归探索最优路劲
            :param _cur_loc_list: 当前已经探索的订单起始地
            :param _rem_loc_list: 当前还没有探索的订单起始地
            :param cur_seats: 当前剩余座位数目
            :param cur_dists: 当前行驶的距离
            """
            if len(_rem_loc_list) == INT_ZERO:
                yield _cur_loc_list.copy(), RouteInfo(True, cur_dists - old_dists, detour_ratios_dict.copy())
            elif is_enough_small((cur_dists - old_dists) * unit_cost, corresponding_optimal_cost):
                for i, o_loc in enumerate(_rem_loc_list):
                    v2o_dist = (network.get_shortest_distance(v_loc, o_loc) if len(_cur_loc_list) == INT_ZERO else network.get_shortest_distance(_cur_loc_list[-1], o_loc))
                    _cur_dists = cur_dists + v2o_dist
                    _order: Order = o_loc.belong_order
                    _cur_loc_list.append(o_loc)
                    if isinstance(o_loc, PickLocation):
                        upper_time = (_order.request_time + _order.wait_time - current_time)
                        if cur_seats - _order.n_riders >= INT_ZERO and network.is_smaller_bound_distance(cur_dists - old_dists, avg_speed * upper_time):
                            # 人数满足要求且不超过最大等待时间
                            _rem_list_copy = _rem_loc_list[:i] + _rem_loc_list[i + 1:]
                            _rem_list_copy.append(_order.drop_location)
                            pick_up_dists_dict[_order] = _cur_dists
                            for n_r, n_i in _recursion(_cur_loc_list, _rem_list_copy, cur_seats - _order.n_riders, _cur_dists):
                                yield n_r, n_i
                            pick_up_dists_dict.pop(_order)

                    else:
                        real_detour_dist = _cur_dists - (pick_up_dists_dict[_order] if _order in pick_up_dists_dict else _order.pick_up_distance) - _order.order_distance
                        if network.is_smaller_bound_distance(FLOAT_ZERO, real_detour_dist) and network.is_smaller_bound_distance(real_detour_dist, _order.detour_distance):
                            # 绕路满足要求
                            _rem_list_copy = _rem_loc_list[:i] + _rem_loc_list[i + 1:]
                            detour_ratios_dict[_order] = (real_detour_dist if real_detour_dist >= FLOAT_ZERO else FLOAT_ZERO) / _order.order_distance
                            for n_r, n_i in _recursion(_cur_loc_list, _rem_list_copy, cur_seats + _order.n_riders, _cur_dists):
                                yield n_r, n_i

                    _cur_loc_list.pop()

        self.reset()  # 优化器初始化！！！！
        if pre_check_need_to_planning(vehicle_type, order, current_time, network):  # 人数都不满足要求不用往后执行
            corresponding_optimal_cost: float = POS_INF
            pick_up_dists_dict: Dict[Order, float] = dict()
            detour_ratios_dict: Dict[Order, float] = dict()
            avg_speed: float = vehicle_type.vehicle_speed
            old_dists: float = vehicle_type.service_driven_distance
            unit_cost: float = vehicle_type.unit_cost
            v_loc: VehicleLocation = vehicle_type.location
            for new_route, new_route_info in _recursion(list(), _generate_remain_loc_list(), vehicle_type.available_seats, vehicle_type.service_driven_distance):
                self._optimizer.optimize(new_route, new_route_info, unit_cost)
                corresponding_optimal_cost = min(new_route_info.route_cost, corresponding_optimal_cost)

        return super(ReschedulingPlanner, self).summary_planning_result()
