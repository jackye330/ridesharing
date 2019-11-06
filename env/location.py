#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/10/28
from typing import NoReturn

__all__ = ["GeoLocation", "OrderLocation", "PickLocation", "DropLocation", "VehicleLocation"]


class GeoLocation:
    """
    地理位置类
    """
    __slots__ = ["osm_index"]

    def __init__(self, osm_index: int):
        """
        :param osm_index: open street map id 字典的索引
        """
        self.osm_index = osm_index

    def __repr__(self):
        return "{0}".format(self.osm_index)

    def __hash__(self):
        return hash(self.osm_index)

    def __eq__(self, other):
        return other.osm_index == self.osm_index


class OrderLocation(GeoLocation):
    __slots__ = ["belong_to_order"]

    def __init__(self, osm_index: int):
        """
        :param osm_index:
        """
        super().__init__(osm_index)

    def set_order_belong(self, order):
        setattr(self, "belong_to_order", order)


class PickLocation(OrderLocation):
    __slots__ = []

    def __init__(self, osm_index: int):
        super().__init__(osm_index)

    def __hash__(self):
        return hash((self.belong_to_order.id, "P"))

    def __eq__(self, other):
        return super().__eq__(other) and isinstance(other, PickLocation)

    def __repr__(self):
        return "({0},{1})".format("PICK", self.osm_index)


class DropLocation(OrderLocation):
    __slots__ = []

    def __init__(self, osm_index: int):
        super().__init__(osm_index)

    def __hash__(self):
        return hash((self.belong_to_order.id, "D"))

    def __eq__(self, other):
        return super().__eq__(other) and isinstance(other, DropLocation)

    def __repr__(self):
        return "({0},{1})".format("DROP", self.osm_index)


class VehicleLocation(GeoLocation):
    """
    由于车辆很有可能行驶到两个节点之间啊，所以车辆的位置始终表示成为
    （地理节点， 一段距离， 下一个地理节点）的形式
    表示车辆处于从当前节点到下一个节点还有行驶一段距离的位置
    """
    __slots__ = ["goal_index", "is_between", "driven_distance", "belong_to_vehicle"]

    def __init__(self, osm_index: int, goal_index=None, is_between=False, driven_distance=0.0):
        super().__init__(osm_index)
        self.goal_index = goal_index
        self.is_between = is_between  # is_between 为True表示车辆在两个节点之间，False表示不再两个节点
        self.driven_distance = driven_distance  # driven_distance 表示车辆在osm_index到goal_index节点之间已经行驶的距离

    def set_vehicle_belong(self, vehicle) -> NoReturn:
        setattr(self, "belong_to_vehicle", vehicle)

    def reset(self) -> NoReturn:
        """
        当车辆不在两个节点之间，而就在一个节点之上的时候就回触发这个函数
        :return:
        """
        self.goal_index = None
        self.is_between = False
        self.driven_distance = 0.0

    def __repr__(self):
        return "VehicleLocation({0})".format(self.osm_index)

    def __hash__(self):
        return hash(self.belong_to_vehicle.vehicle_id)

    def __eq__(self, other):
        return self.belong_to_vehicle.vehicle_id == other.belong_to_vehicle.vehicle_id


if __name__ == '__main__':
    p1 = PickLocation(1)
    p2 = PickLocation(2)
    order1 = 1
    order2 = 2
    p1.set_order_belong(order1)
    p2.set_order_belong(order2)

    d1 = DropLocation(1)
    d2 = DropLocation(4)
    d1.set_order_belong(order1)
    d2.set_order_belong(order2)
    print(isinstance(d1, OrderLocation))


    def f(p: OrderLocation):
        print(isinstance(p, PickLocation))


    f(d1)
    print(p1 == p2)
    # print(hash(p1))
    # print(hash(d1))
