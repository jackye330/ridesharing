#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/3/21
import pickle

import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Times New Roman', weight=3)
model1 = "Nearest-Matching"
model2 = "SWMOM-VCG"
model3 = "SWMOM-GM"

min_x = 500
max_x = 2501
step_x = 500
n_sample = 10
legend_fontsize = 14
label_fontsize = 16

plt.show()
sw1_mean, sw1_std = [], []
sw2_mean, sw2_std = [], []
sw3_mean, sw3_std = [], []
sc1_mean, sc1_std = [], []
sc2_mean, sc2_std = [], []
sc3_mean, sc3_std = [], []
u1_mean, u1_std = [], []
u2_mean, u2_std = [], []
u3_mean, u3_std = [], []
p1_mean, p1_std = [], []
p2_mean, p2_std = [], []
p3_mean, p3_std = [], []
ratio1_mean, ratio1_std = [], []
ratio2_mean, ratio2_std = [], []
ratio3_mean, ratio3_std = [], []
for v in range(min_x, max_x, step_x):
    t_sw1 = []
    t_sw2 = []
    t_sw3 = []
    t_sc1 = []
    t_sc2 = []
    t_sc3 = []
    t_p1 = []
    t_p2 = []
    t_p3 = []
    t_u1 = []
    t_u2 = []
    t_u3 = []
    r1 = []
    r2 = []
    r3 = []
    for k in range(n_sample):
        with open("./result/{0}/{1}_{2}.pkl".format(model1, v, k), "rb") as file:
            result1 = pickle.load(file)
            social_welfare_result1, social_cost_result1, _, total_utility_result1, total_profit_result1, service_rate1 = result1
        with open("./result/{0}/{1}_{2}.pkl".format(model2, v, k), "rb") as file:
            result2 = pickle.load(file)
            social_welfare_result2, social_cost_result2, _, total_utility_result2, total_profit_result2, service_rate2 = result2
        with open("./result/{0}/{1}_{2}.pkl".format(model3, v, k), "rb") as file:
            result3 = pickle.load(file)
            social_welfare_result3, social_cost_result3, _, total_utility_result3, total_profit_result3, service_rate3 = result3

        t_sw1.append(np.sum(social_welfare_result1))
        t_sw2.append(np.sum(social_welfare_result2))
        t_sw3.append(np.sum(social_welfare_result3))
        t_sc1.append(np.sum(social_cost_result1))
        t_sc2.append(np.sum(social_cost_result2))
        t_sc3.append(np.sum(social_cost_result3))
        r1.append(service_rate1)
        r2.append(service_rate2)
        r3.append(service_rate3)
        t_p1.append(np.sum(total_profit_result1))
        t_p2.append(np.sum(total_profit_result2))
        t_p3.append(np.sum(total_profit_result3))
        t_u1.append(np.sum(total_utility_result1))
        t_u2.append(np.sum(total_utility_result2))
        t_u3.append(np.sum(total_utility_result3))

    sw1_mean.append(np.mean(t_sw1))
    sw1_std.append(np.std(t_sw1))
    sw2_mean.append(np.mean(t_sw2))
    sw2_std.append(np.std(t_sw2))
    sw3_mean.append(np.mean(t_sw3))
    sw3_std.append(np.std(t_sw3))
    sc1_mean.append(np.mean(t_sc1))
    sc1_std.append(np.std(t_sc1))
    sc2_mean.append(np.mean(t_sc2))
    sc2_std.append(np.std(t_sc2))
    sc3_mean.append(np.mean(t_sc3))
    sc3_std.append(np.std(t_sc3))
    p1_mean.append(np.mean(t_p1))
    p1_std.append(np.std(t_p1))
    p2_mean.append(np.mean(t_p2))
    p2_std.append(np.std(t_p2))
    p3_mean.append(np.mean(t_p3))
    p3_std.append(np.std(t_p3))
    u1_mean.append(np.mean(t_u1))
    u1_std.append(np.std(t_u1))
    u2_mean.append(np.mean(t_u2))
    u2_std.append(np.std(t_u2))
    u3_mean.append(np.mean(t_u3))
    u3_std.append(np.std(t_u3))
    ratio1_mean.append(np.mean(r1))
    ratio1_std.append(np.std(r1))
    ratio2_mean.append(np.mean(r2))
    ratio2_std.append(np.std(r2))
    ratio3_mean.append(np.mean(r3))
    ratio3_std.append(np.std(r3))


def plot_bar_char1(y1_mean, y1_std, x_str, y_str, model_1):
    plt.figure(figsize=(7, 6))
    ind = np.arange(len(y1_mean)) * 2
    label = range(min_x, max_x, step_x)
    width = 0.8
    plt.bar(ind, y1_mean, width=width, yerr=y1_std)
    plt.xticks(ind, label)
    plt.xlabel(x_str, fontsize=label_fontsize)
    plt.ylabel(y_str, fontsize=label_fontsize)
    plt.legend(model_1, fontsize=legend_fontsize)
    plt.show()


def plot_bar_char2(y1_mean, y2_mean, y1_std, y2_std, x_str, y_str, model_1, model_2):
    plt.figure(figsize=(7, 6))
    ind = np.arange(len(y1_mean)) * 2
    label = range(min_x, max_x, step_x)
    width = 0.8
    plt.bar(ind - width / 2, y1_mean, width=width, yerr=y1_std)
    plt.bar(ind + width / 2, y2_mean, width=width, yerr=y2_std)
    plt.xticks(ind, label)
    plt.legend([model_1, model_2], fontsize=legend_fontsize)
    plt.xlabel(x_str, fontsize=label_fontsize)
    plt.ylabel(y_str, fontsize=label_fontsize)
    plt.show()


def plot_bar_char3(y1_mean, y2_mean, y3_mean, y1_std, y2_std, y3_std, x_str, y_str, model_1, model_2, model_3):
    plt.figure(figsize=(7, 6))
    ind = np.arange(len(y1_mean)) * 2
    width = 0.6
    plt.bar(ind - width, y1_mean, width=width, yerr=y1_std)
    plt.bar(ind, y2_mean, width=width, yerr=y2_std)
    plt.bar(ind + width, y3_mean, width=width, yerr=y3_std)
    plt.xticks(ind, range(min_x, max_x, step_x))
    plt.legend([model_1, model_2, model_3], fontsize=legend_fontsize)
    plt.xlabel(x_str, fontsize=label_fontsize)
    plt.ylabel(y_str, fontsize=label_fontsize)
    plt.show()


def plot_trend_char2(y1_mean, y2_mean, x_str, y_str, model_1, model_2):
    plt.figure(figsize=(7, 6))
    ind = range(len(y1_mean))
    label = range(min_x, max_x, step_x)
    plt.plot(ind, y1_mean, 'r-s', linewidth=3, markersize=10, markerfacecolor='w', markeredgewidth=3)
    plt.plot(ind, y2_mean, 'b-^', linewidth=3, markersize=10, markerfacecolor='w', markeredgewidth=3)
    plt.xticks(ind, label)
    plt.legend([model_1, model_2], fontsize=legend_fontsize)
    plt.grid(True)
    plt.xlabel(x_str, fontsize=label_fontsize)
    plt.ylabel(y_str, fontsize=label_fontsize)
    plt.show()


def plot_trend_char3(y1_mean, y2_mean, y3_mean, x_str, y_str, model_1, model_2, model_3):
    plt.figure(figsize=(7, 6))
    ind = range(len(y1_mean))
    label = range(min_x, max_x, step_x)
    plt.errorbar(ind, y1_mean, fmt='g-v', linewidth=3, markersize=10, markerfacecolor='w', markeredgewidth=3)
    plt.errorbar(ind, y2_mean, fmt='r-s', linewidth=3, markersize=10, markerfacecolor='w', markeredgewidth=3)
    plt.errorbar(ind, y3_mean, fmt='b-^', linewidth=3, markersize=10, markerfacecolor='w', markeredgewidth=3)
    plt.xticks(ind, label)
    plt.legend([model_1, model_2, model_3], fontsize=legend_fontsize)
    plt.grid(True)
    plt.xlabel(x_str, fontsize=label_fontsize)
    plt.ylabel(y_str, fontsize=label_fontsize)
    plt.show()


plot_trend_char3(sw1_mean, sw2_mean, sw3_mean, "#Vehicles", "Social Welfare", model1, model2,
                 model3)
plot_trend_char3(ratio1_mean, ratio2_mean, ratio3_mean, "#Vehicles",
                 "Ratio of Served Orders", model1, model2, model3)

plot_trend_char3(sc1_mean, sc2_mean, sc3_mean, "#Vehicles", "Social Cost", model1, model2, model3)

plot_trend_char2(p2_mean, p3_mean, "#Vehicles", "Profit of the Platform", model2, model3)
plot_trend_char2(u2_mean, u3_mean, "#Vehicles", "Total Payoff of Drivers", model2, model3)
au1_mean = np.array(u1_mean) / np.arange(min_x, max_x, step_x)
au1_std = np.array(u1_std) / np.arange(min_x, max_x, step_x)
au2_mean = np.array(u2_mean) / np.arange(min_x, max_x, step_x)
au2_std = np.array(u2_std) / np.arange(min_x, max_x, step_x)
au3_mean = np.array(u3_mean) / np.arange(min_x, max_x, step_x)
au3_std = np.array(u3_std) / np.arange(min_x, max_x, step_x)
plot_trend_char2(au2_mean, au3_mean, "#Vehicles", "Average Payoff of Drivers", model2, model3)
