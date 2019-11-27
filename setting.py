#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/4
import numpy as np

# 将一些常数实现为单例，节约空间
VALUE_EPS = 1E-8  # 浮点数相等的最小精度
FLOAT_ZERO = 0.0
INT_ZERO = 0
FIRST_INDEX = INT_ZERO
POS_INF = np.PINF
NEG_INF = np.NINF
MILE_TO_KM = 1.609344
MILE_TO_M = 1609.344
SECOND_OF_DAY = 86_400  # 24 * 60 * 60 一天有多少秒

# real_transport 和 grid_transport 实验都可以使用的数据 ######################################################################################################################
# 实验得环境 为"REAL_TRANSPORT"表示真实的路网环境 为"GRID_TRANSPORT"表示虚拟网格环境
REAL_TRANSPORT = "REAL_TRANSPORT"
GRID_TRANSPORT = "GRID_TRANSPORT"
EXPERIMENTAL_MODE = REAL_TRANSPORT
# 一组参数实验的重复次数
MAX_REPEATS = 10
# 订单分配算法的执行时间间隔 单位 s. 如果是路网环境 [10 15 20 25 30], 如果是网格环境 默认为1.
TIME_SLOT = 25
# 距离精度误差, 表示一个车辆到某一个点的距离小于这一个数, 那么就默认这个车已经到这个点上了 单位 m. 如果是实际的路网一般取10.0m, 如果是网格环境一般取0.0.
DISTANCE_EPS = 10.0
# 模拟天数的最小值/最大值，如果是网格环境默认为0, 如果是网格环境默认为1.
MIN_REQUEST_DAY, MAX_REQUEST_DAY = 0, 1
# 模拟一天的时刻最小值/最大值 单位 s.
# 如果是路网环境 MIN_REQUEST_TIME <= request_time < MAX_REQUEST_TIME 并且有 MAX_REQUEST_TIME - MIN_REQUEST_TIME 并且可以整除 TIME_SLOT.
# 如果是网格环境 MIN_REQUEST_TIME = 0, MIN_REQUEST_TIME = 500.
MIN_REQUEST_TIME, MAX_REQUEST_TIME = 8 * 60 * 60, 9 * 60 * 60
# 实验环境中的车辆数目
VEHICLE_NUMBER = 200
# 实验环境中的车辆速度 单位 m/s. 对于任意的环境 VEHICLE_SPEED * TIME_SLOT >> DISTANCE_EPS. 纽约市规定是 MILE_TO_KM * 12 / 3.6 m/s
VEHICLE_SPEED = MILE_TO_KM * 12 / 3.6
# 投标策略 "ADDITIONAL_COST" 以成本量的增加量作为投标 "ADDITIONAL_PROFIT" 以利润的增加量作为投标量
ADDITIONAL_COST_STRATEGY = "ADDITIONAL_COST_STRATEGY"
ADDITIONAL_PROFIT_STRATEGY = "ADDITIONAL_PROFIT_STRATEGY"
BIDDING_STRATEGY = ADDITIONAL_PROFIT_STRATEGY
# 路线规划的目标 "MINIMIZE_COST" 最小化成本 "MAXIMIZE_PROFIT" 最大化利润
MINIMIZE_WAIT_TIME = "MINIMIZE_WAIT_TIME"
MINIMIZE_COST = "MINIMIZE_COST"
MAXIMIZE_PROFIT = "MAXIMIZE_PROFIT"
ROUTE_PLANNING_GOAL = MAXIMIZE_PROFIT
# 路线规划的方案 "INSERTING" 新的订单起始点直接插入而不改变原有订单起始位置顺序  "RESCHEDULING" 原有订单的起始位置进行重排
INSERTING = "INSERTING"
RESCHEDULING = "RESCHEDULING"
ROUTE_PLANNING_METHOD = INSERTING
# 平台使用的订单分发方式
NEAREST_DISPATCHING = "NEAREST-DISPATCHING"  # 通用的最近车辆分配算法
VCG_MECHANISM = "SWMOM-VCG"  # vcg 机制 这是一个简单的分配机制
GM_MECHANISM = "SWMOM-GM"  # gm 机制 这是一个简单的分配机制
SPARP_MECHANISM = "SPARP"  # SPARP 机制 这是一个通用分配机制
SEQUENCE_AUCTION = "SWMOM-SASP"  # 贯序拍卖机制 这是一个通用分配机制
DISPATCHING_METHOD = SPARP_MECHANISM

# 与 REAL 相关的配置 ###################################################################################################################################
# 与地理相关的数据存放点
HaiKou = "HaiKou"
Manhattan = "Manhattan"
GEO_NAME = Manhattan
GEO_DATA_FILE = {
    "base_folder": "./data/{0}/network_data".format(GEO_NAME),
    "graph_file": "{0}.graphml".format(GEO_NAME),
    "osm_id2index_file": "osm_id2index.pkl",
    "index2osm_id_file": "index2osm_id.pkl",
    "shortest_distance_file": "shortest_distance.npy",
    "shortest_path_file": "shortest_path.npy",
    "adjacent_index_file": "adjacent_index.pkl",
    "access_index_file": "access_index.pkl",
    "adjacent_location_osm_index_file": "{0}/adjacent_location_osm_index.pkl".format(TIME_SLOT),
    "adjacent_location_driven_distance_file": "{0}/adjacent_location_driven_distance.pkl".format(TIME_SLOT),
    "adjacent_location_goal_index_file": "{0}/adjacent_location_goal_index.pkl".format(TIME_SLOT),
}
# 订单数据存放地址
ORDER_DATA_FILES = ["./data/{0}/order_data/order_data_{1:03}.csv".format(GEO_NAME, day) for day in range(MIN_REQUEST_DAY, MAX_REQUEST_DAY)]
# 车辆油耗与座位数据存放地址
FUEL_CONSUMPTION_DATA_FILE = "./data/vehicle_data/fuel_consumption_and_seats.csv"
# 直接与此常数相乘可以得到单位距离的成本 $/m/(单位油耗)
VEHICLE_FUEL_COST_RATIO = 2.5 / 6.8 / MILE_TO_M
# 乘客最大绕路比可选范围
DETOUR_RATIOS = [0.25, 0.50, 0.75, 1.00]
# 乘客最大等待时间可选范围 单位 s
WAIT_TIMES = [3 * 60, 4 * 60, 5 * 60, 6 * 60, 7 * 60, 8 * 60]

# GRID 相关的设定 ######################################################################################################################################
# 网格的规模，横向/纵向的网格数目
GRAPH_SIZE = 100
# 每一个网格的大小
GRID_SIZE = 1.0
# 每一个时间槽我们的订单数目我们按照正态分布生成
MU, SIGMA = 100, 10
# 订单的等待时间
MIN_WAIT_TIME, MAX_WAIT_TIME = 10, 100
# 车辆成本单位成本可选范围 每米司机要花费的钱
UNIT_COSTS = [1.2, 1.3, 1.4, 1.5]
# 订单的单位预先价格 每米乘客要花费的钱
UNIT_FARE = 2.5
# 车辆空余位置的选择范围
N_SEATS = 4
# 订单的乘客数目
MIN_N_RIDERS, MAX_N_RIDERS = 1, 2

# 与环境创建相关的数据 #################################################################
INPUT_VEHICLES_DATA_FILES = ["./data/input/vehicles_data/{0}_{1}_{2}_{3}.csv".format(EXPERIMENTAL_MODE, i, VEHICLE_NUMBER, TIME_SLOT) for i in range(MAX_REPEATS)]
INPUT_ORDERS_DATA_FILES = ["./data/input/orders_data/{0}_{1}_{2}_{3}.csv".format(EXPERIMENTAL_MODE, i, VEHICLE_NUMBER, TIME_SLOT) for i in range(MAX_REPEATS)]
SAVE_RESULT_FILES = ["./result/{0}/{1}_{2}_{3}_{4}_{5}_{6}.pkl".format(DISPATCHING_METHOD, EXPERIMENTAL_MODE, i, VEHICLE_NUMBER, TIME_SLOT, MIN_REQUEST_TIME, MAX_REQUEST_TIME) for i in range(MAX_REPEATS)]
