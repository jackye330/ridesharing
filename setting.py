#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/2/21
# import sys
# import setting
#
#
# class _setting(object):
#     class ConstError(TypeError):
#         pass
#
#     class ConstCaseError(ConstError):
#         pass
#
#     def __setattr__(self, key, value):
#         if key in self.__dict__:
#             raise self.ConstError("Can't change const %s" % key)
#         if not key.isupper():
#             raise self.ConstCaseError("const name %s is not all uppercase" % key)
#         self.__dict__[key] = value
#
#
# sys.modules[__name__] = _setting

# 超参数
VEHICLE_NUMBER = 500  # 车辆数目
AVERAGE_SPEED = 1.609344 * 12 / 3.6  # 汽车的速度 单位 m/s  AVERAGE_SPEED * TIME_SLOT >> 10.0
TIME_SLOT = 60  # 订单分配算法的时间间隔 单位 s
DISTANCE_EPS = 10.0  # 距离进度误差，表示一个车辆到某一个点的距离小于这一个数，那么就默认这个车已经到这个点上了 单位 m
DETOUR_RATIOS = [0.25, 0.50, 0.75, 1.00]  # 乘客绕路比可选范围
MAX_WAIT_TIMES = [3 * 60, 4 * 60, 5 * 60, 6 * 60, 7 * 60, 8 * 60]  # 乘客最大等待时间 单位 s
DRIVER_FARES = [1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.80, 1.85, 1.90]  # single minded
DRIVER_FUEL_COST_RATIO = 2.5 / 1.609344 / 6.8 / 1000  # 直接与此常数相乘可以得到单位距离的成本
MIN_REQUEST_TIME = 0
MAX_REQUEST_TIME = 2 * 60 * 60  # MIN_REQUEST_TIME <= request_time < MAX_REQUEST_TIME
MIN_REPEATS = 0   # 最小模拟次数
MAX_REPEATS = 10  # 最大模拟次数
