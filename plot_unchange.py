#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/20
import matplotlib.pyplot as plt

import pickle

with open("./500_0.pkl", "rb") as file:
	data = pickle.load(file)

social_welfare_result, social_cost_result, total_payment_result, total_utility_result, total_profit_result, unchange_vehicle_result, empty_vehicle_rate, service_rate = data
# fig, ax = plt.subplots(1, 1)
# ax_sub = ax.twinx()
# ax_sub.plot(range(1440), total_payment_result)
# ax_sub.plot(range(1440), empty_vehicle_rate)
# ax_sub.set_ylabel("unchange rate")
# ax_sub.set_ylabel("driver payment")
# ax.set_xlabel('time')
# plt.show()
plt.plot(empty_vehicle_rate)
plt.show()
# print(service_rate)
print(sum(total_payment_result))
print(sum(empty_vehicle_rate))
print(sum(total_payment_result) / (98235 * service_rate))
