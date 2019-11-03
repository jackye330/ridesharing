# #!/usr/bin/env python3
# # -*- coding:utf-8 -*-
# # author : zlq16
# # date   : 2019/6/3
import pickle
from algorithm import *
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from agent import GeoLocation, OrderLocation, Order, Vehicle

row_size = 100
col_size = 100
xs = list(range(row_size))
ys = list(range(col_size))
vehicle_size = 10
order_size = 10
max_wait_time = 20
detour_ratio = 0.5

vehicle_capacity = 4
average_speed = 1


#### social welfare
# 直接使用曼哈顿距离
def distance(point1, point2):
    return np.abs(point1[0] - point2[0]) + np.abs(point1[1] - point2[1])


def generate_random_order(order_id, request_time):
    request_x = np.random.choice(xs)
    request_y = np.random.choice(ys)
    while True:
        end_x = np.random.choice(xs)
        end_y = np.random.choice(ys)
        if end_x != request_x and end_y != request_y:
            break
    start_location = OrderLocation(request_x * 10 + request_y, request_x, request_y, OrderLocation.PICK_UP_TYPE)
    end_location = OrderLocation(end_x * 10 + end_y, end_x, end_y, OrderLocation.DROP_OFF_TYPE)
    order_distance = distance((request_x, request_x), (end_x, end_y))
    fare = 2 * order_distance
    detour_ratio = np.random.choice([0.25, 0.5, 0.75, 1.0])
    order = Order(order_id, start_location, end_location, request_time, max_wait_time,
                  order_distance, fare, detour_ratio, 1)
    return order


def generate_random_vehicle(vehicle_id):
    x = np.random.choice(xs)
    y = np.random.choice(ys)
    location = GeoLocation(x * row_size + y, x, y)
    cost_pre_km = np.random.uniform(1.0, 1.5)
    vehicle = Vehicle(vehicle_id, location, vehicle_capacity, cost_pre_km, status=Vehicle.WITHOUT_MISSION_STATUS)
    return vehicle


Vehicle.set_average_speed(average_speed)
x = [1, 2, 3, 4, 5]
# sw, sr, p, ap = [], [], [], []
# for vehicle_size in range(100, 200, 100):
#     for order_size in range(100, 501, 100):
#         info1, info2, info3 = [], [], []
#         for k in range(10):
#             vehicles = [generate_random_vehicle(i) for i in range(vehicle_size)]
#             orders = set([generate_random_order(i, 0) for i in range(order_size)])
#
#             result1 = dispatch_orders_with_nearest_vehicle(row_size, col_size, deepcopy(orders), deepcopy(vehicles), 0,
#                                                            True)
#             dispatch_orders, payments, social_welfare1, _, total_payment1, _, _ = result1
#             info1.append([social_welfare1, sum([len(payments[v]) for v in payments]) / order_size])
#
#             result2 = dispatch_orders_with_vcg(row_size, col_size, deepcopy(orders), deepcopy(vehicles), 0, True)
#             dispatch_orders, payments, social_welfare2, _, total_payment2, _, _ = result2
#             info2.append([social_welfare2, len(payments) / order_size])
#
#             result3 = dispatch_orders_with_greedy(row_size, col_size, deepcopy(orders), deepcopy(vehicles), 0, True)
#             dispatch_orders, payments, social_welfare3, _, total_payment3, _, _ = result3
#             info3.append([social_welfare3, sum([len(payments[v]) for v in payments]) / order_size])
#
#         info1 = np.array(info1).mean(axis=0)
#         info2 = np.array(info2).mean(axis=0)
#         info3 = np.array(info3).mean(axis=0)
#         print(info1)
#         sw.append([info1[0], info2[0], info3[0]])
#         sr.append([info1[1], info2[1], info3[1]])

with open("./result/fix_vehicle.pkl", "rb") as f:
    sw, sr = pickle.load(f)
sw = np.array(sw)
sr = np.array(sr)
plt.figure(figsize=(8, 7))
plt.rc('font', family='Times New Roman', weight=3)
plt.plot(x, sw[:, 0], 'r-D', linewidth=3, markersize=10)
plt.plot(x, sw[:, 1], 'b-^', linewidth=3, markersize=10)
plt.plot(x, sw[:, 2], 'g-v', linewidth=3, markersize=10)
plt.xticks(x, [100, 200, 300, 400, 500])
plt.xlabel("#Orders", fontsize=18)
plt.ylabel("Social Welfare", fontsize=18)
plt.legend(["Nearest-Matching", "SWMOM-VCG", "MSWCRR-GREEDY"], fontsize=18)
plt.grid(True)
plt.show()
plt.figure(figsize=(8, 7))
plt.plot(x, sr[:, 0], 'r-d', linewidth=3, markersize=10)
plt.plot(x, sr[:, 1], 'b-^', linewidth=3, markersize=10)
plt.plot(x, sr[:, 2], 'g-v', linewidth=3, markersize=10)
plt.xticks(x, [100, 200, 300, 400, 500])
plt.xlabel("#Orders", fontsize=18)
plt.ylabel("Ratio of Served Orders", fontsize=18)
plt.legend(["Nearest-Matching", "SWMOM-VCG", "MSWCRR-GREEDY"], fontsize=18)
plt.grid(True)
plt.show()
# with open("./result/fix_vehicle.pkl", "wb") as f:
#     pickle.dump((sw, sr), f)
