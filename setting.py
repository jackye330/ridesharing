#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/2/21
# import sys
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
# import setting

# 超参数
VEHICLE_NUMBER = 500
DETOUR_RATIOS = [0.25, 0.50, 0.75, 1.00]
MAX_WAIT_TIMES = [3, 4, 5, 6, 7, 8]
DRIVER_FARES = [1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.80, 1.85, 1.90]  # single minded
AVERAGE_SPEED = 1.609344 * 12 / 60  # km/min
MIN_REQUEST_TIME = 0
MAX_REQUEST_TIME = 1440  # MIN_REQUEST_TIME <= request_time < MAX_REQUEST_TIME
MIN_REPEATS = 0  # 最小模拟次数
MAX_REPEATS = 1  # 最大模拟次数
