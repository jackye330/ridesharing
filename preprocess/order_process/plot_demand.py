#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/12/9
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# f, ax = plt.subplots(5, 5)
# weekends = {3, 4, 10, 11, 17, 18, 24, 25}
# weekday = set(range(30)) - weekends
# for i, day in enumerate(weekday):
#     r, c = i // 5, i % 5
#     df = pd.read_csv("../../data/Manhattan/order_data/order_data_{:03d}.csv".format(day))
#     ax[r][c].hist(df.pick_time.values)
#
# plt.show()

plt.figure()
demand_model = np.load("../../data/Manhattan/order_data/order_model/demand_model.npy")
plt.plot(range(len(demand_model)), demand_model, 'r-d', linewidth=3, markersize=10, markerfacecolor='k', markeredgewidth=3)
plt.xlim([0, 23])
plt.xlabel("Hour of Day")
plt.ylabel("Number of Query")
plt.grid(True)
plt.show()

plt.figure()
unit_fare = np.load("../../data/Manhattan/order_data/order_model/unit_fare_model.npy")
plt.plot(range(len(unit_fare)), unit_fare, 'r-d', linewidth=3, markersize=10, markerfacecolor='k', markeredgewidth=3)
plt.xlabel("Hour of Day")
plt.ylabel("Unit Fare of Day ($/mile)")
plt.xlim([0, 23])

plt.grid(True)
plt.show()

plt.figure()
demand_location_model = np.load("../../data/Manhattan/order_data/order_model/demand_location_model.npy")
plt.plot(range(len(demand_location_model[0])), demand_location_model[0], "r-d", linewidth=3, markersize=10, markerfacecolor='k', markeredgewidth=3)
plt.plot(range(len(demand_location_model[23])), demand_location_model[23], "b-d", linewidth=3, markersize=10, markerfacecolor='k', markeredgewidth=3)
plt.xlabel("Pick up Region of Id")
plt.grid(True)
plt.show()

plt.figure()
demand_transfer_model = np.load("../../data/Manhattan/order_data/order_model/demand_transfer_model.npy")
drop_transfer = np.multiply(demand_transfer_model[0], demand_location_model[0].reshape(len(demand_location_model[0]), 1))
drop_transfer = np.sum(drop_transfer, axis=0)
plt.plot(range(len(drop_transfer)), drop_transfer, "r-o",  linewidth=3, markersize=10, markerfacecolor='k', markeredgewidth=3)
drop_transfer = np.multiply(demand_transfer_model[1], demand_location_model[1].reshape(len(demand_location_model[1]), 1))
drop_transfer = np.sum(drop_transfer, axis=0)
plt.plot(range(len(drop_transfer)), drop_transfer, "b-o",  linewidth=3, markersize=10, markerfacecolor='k', markeredgewidth=3)
plt.xlabel("Drop off Region of Id")
plt.grid(True)
plt.show()