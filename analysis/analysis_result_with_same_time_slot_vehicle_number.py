#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/3/21
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from analysis.utility import Result
import pandas as pd
from setting import GM_MECHANISM, VCG_MECHANISM
from setting import EXPERIMENTAL_MODE
from setting import MIN_REQUEST_TIME, MAX_REQUEST_TIME

matplotlib.use('TkAgg')
plt.rc('font', family='Times New Roman', weight=3)
legend_font_size = 14
label_font_size = 16
n_sample = 10
index2metric = pd.read_csv("index2metric.csv")
model_names = [VCG_MECHANISM, GM_MECHANISM]
min_v, max_v, v_step = 800, 1201, 200


def plot_bar_char2(y1_mean, y2_mean, y1_std, y2_std, x_str, y_str, model_1, model_2):
    plt.figure(figsize=(7, 6))
    ind = np.arange(len(y1_mean)) * 2
    label = range(min_v, max_v, v_step)
    width = 0.8
    plt.bar(ind - width / 2, y1_mean, width=width, yerr=y1_std)
    plt.bar(ind + width / 2, y2_mean, width=width, yerr=y2_std)
    plt.xticks(ind, label)
    plt.legend([model_1, model_2], fontsize=legend_font_size)
    plt.xlabel(x_str, fontsize=label_font_size)
    plt.ylabel(y_str, fontsize=label_font_size)
    plt.show()


def get_each_parameter_value(mechanism, vehicle_number, time_slot):
    sw, dp, pp, sr, rt = [], [], [], [], []
    for i in range(n_sample):
        file_name = "../result/{0}/{1}_{2}_{3}_{4}_{5}_{6}.pkl".format(mechanism, EXPERIMENTAL_MODE, i, vehicle_number, time_slot, MIN_REQUEST_TIME, MAX_REQUEST_TIME)
        with open(file_name, "rb") as file:
            data = pickle.load(file)
        r = Result(data)
        sw.append(r.get_total_social_welfare())
        dp.append(r.get_total_driver_payoffs())
        pp.append(r.get_total_platform_profit())
        sr.append(r.get_service_ratio())
        rt.append(r.get_total_running_time())

    return (
        np.mean(sw), np.std(sw),
        np.mean(dp), np.std(dp),
        np.mean(pp), np.std(pp),
        np.mean(sr), np.std(sr),
        np.mean(rt), np.std(rt))


# vcg 数据
vcg_sw_mean, vcg_sw_std = [], []
vcg_dp_mean, vcg_dp_std = [], []
vcg_pp_mean, vcg_pp_std = [], []
vcg_sr_mean, vcg_sr_std = [], []
vcg_rt_mean, vcg_rt_std = [], []
for v in range(min_v, max_v, v_step):
    print(v)
    summary_result = get_each_parameter_value(VCG_MECHANISM, v, 30)
    vcg_sw_mean.append(summary_result[0])
    vcg_sw_std.append(summary_result[1])
    vcg_dp_mean.append(summary_result[2])
    vcg_dp_std.append(summary_result[3])
    vcg_pp_mean.append(summary_result[4])
    vcg_pp_std.append(summary_result[5])
    vcg_sr_mean.append(summary_result[6])
    vcg_sr_std.append(summary_result[7])
    vcg_rt_mean.append(summary_result[8])
    vcg_rt_std.append(summary_result[9])


# gm 数据
gm_sw_mean, gm_sw_std = [], []
gm_dp_mean, gm_dp_std = [], []
gm_pp_mean, gm_pp_std = [], []
gm_sr_mean, gm_sr_std = [], []
gm_rt_mean, gm_rt_std = [], []
for v in range(min_v, max_v, v_step):
    summary_result = get_each_parameter_value(GM_MECHANISM, v, 30)
    gm_sw_mean.append(summary_result[0])
    gm_sw_std.append(summary_result[1])
    gm_dp_mean.append(summary_result[2])
    gm_dp_std.append(summary_result[3])
    gm_pp_mean.append(summary_result[4])
    gm_pp_std.append(summary_result[5])
    gm_sr_mean.append(summary_result[6])
    gm_sr_std.append(summary_result[7])
    gm_rt_mean.append(summary_result[8])
    gm_rt_std.append(summary_result[9])


plot_bar_char2(vcg_sw_mean, gm_sw_mean, vcg_sw_std, gm_sw_std, "#vehilce", "social welfare", VCG_MECHANISM, GM_MECHANISM)
plot_bar_char2(vcg_sr_mean, gm_sr_mean, vcg_sr_std, gm_sr_std, "#vehilce", "service ratio", VCG_MECHANISM, GM_MECHANISM)



# sw1_mean, sw1_std = [], []
# sw2_mean, sw2_std = [], []
# sw3_mean, sw3_std = [], []
# sw4_mean, sw4_std = [], []
# sw5_mean, sw5_std = [], []
# sc1_mean, sc1_std = [], []
# sc2_mean, sc2_std = [], []
# sc3_mean, sc3_std = [], []
# sc4_mean, sc4_std = [], []
# sc5_mean, sc5_std = [], []
# u1_mean, u1_std = [], []
# u2_mean, u2_std = [], []
# u3_mean, u3_std = [], []
# u4_mean, u4_std = [], []
# u5_mean, u5_std = [], []
# p1_mean, p1_std = [], []
# p2_mean, p2_std = [], []
# p3_mean, p3_std = [], []
# p4_mean, p4_std = [], []
# p5_mean, p5_std = [], []
# ratio1_mean, ratio1_std = [], []
# ratio2_mean, ratio2_std = [], []
# ratio3_mean, ratio3_std = [], []
# ratio4_mean, ratio4_std = [], []
# ratio5_mean, ratio5_std = [], []
# for v in range(min_x, max_x, step_x):
#     t_sw1 = []
#     t_sw2 = []
#     t_sw3 = []
#     t_sw4 = []
#     t_sw5 = []
#     t_sc1 = []
#     t_sc2 = []
#     t_sc3 = []
#     t_sc4 = []
#     t_sc5 = []
#     t_p1 = []
#     t_p2 = []
#     t_p3 = []
#     t_p4 = []
#     t_p5 = []
#     t_u1 = []
#     t_u2 = []
#     t_u3 = []
#     t_u4 = []
#     t_u5 = []
#     r1 = []
#     r2 = []
#     r3 = []
#     r4 = []
#     r5 = []
#     for k in range(n_sample):
#         with open("./result/{0}/{1}_{2}.pkl".format(model1, v, k), "rb") as file:
#             result1 = pickle.load(file)
#             social_welfare_result1, social_cost_result1, _, total_utility_result1, total_profit_result1, service_rate1 = result1
#         with open("./result/{0}/{1}_{2}.pkl".format(model2, v, k), "rb") as file:
#             result2 = pickle.load(file)
#             social_welfare_result2, social_cost_result2, _, total_utility_result2, total_profit_result2, service_rate2 = result2
#         with open("./result/{0}/{1}_{2}.pkl".format(model3, v, k), "rb") as file:
#             result3 = pickle.load(file)
#             social_welfare_result3, social_cost_result3, _, total_utility_result3, total_profit_result3, service_rate3 = result3
#         with open("./result/{0}/{1}_{2}.pkl".format(model4, v, k), "rb") as file:
#             result4 = pickle.load(file)
#             social_welfare_result4, social_cost_result4, _, total_utility_result4, total_profit_result4, service_rate4 = result4
#         t_sw1.append(np.sum(social_welfare_result1))
#         t_sw2.append(np.sum(social_welfare_result2))
#         t_sw3.append(np.sum(social_welfare_result3))
#         t_sw4.append(np.sum(social_welfare_result4))
#         t_sw4.append(np.sum(social_welfare_result4))
#
#         t_sc1.append(np.sum(social_cost_result1))
#         t_sc2.append(np.sum(social_cost_result2))
#         t_sc3.append(np.sum(social_cost_result3))
#         t_sc4.append(np.sum(social_cost_result4))
#
#         r1.append(service_rate1)
#         r2.append(service_rate2)
#         r3.append(service_rate3)
#         r4.append(service_rate4)
#
#         t_p1.append(np.sum(total_profit_result1))
#         t_p2.append(np.sum(total_profit_result2))
#         t_p3.append(np.sum(total_profit_result3))
#         t_p4.append(np.sum(total_profit_result4))
#
#         t_u1.append(np.sum(total_utility_result1))
#         t_u2.append(np.sum(total_utility_result2))
#         t_u3.append(np.sum(total_utility_result3))
#         t_u4.append(np.sum(total_utility_result4))
#
#     sw1_mean.append(np.mean(t_sw1))
#     sw1_std.append(np.std(t_sw1))
#     sw2_mean.append(np.mean(t_sw2))
#     sw2_std.append(np.std(t_sw2))
#     sw3_mean.append(np.mean(t_sw3))
#     sw3_std.append(np.std(t_sw3))
#     sw4_mean.append(np.mean(t_sw4))
#     sw4_std.append(np.std(t_sw4))
#     sw5_mean.append(np.mean(t_sw5))
#     sw5_std.append(np.std(t_sw5))
#
#     sc1_mean.append(np.mean(t_sc1))
#     sc1_std.append(np.std(t_sc1))
#     sc2_mean.append(np.mean(t_sc2))
#     sc2_std.append(np.std(t_sc2))
#     sc3_mean.append(np.mean(t_sc3))
#     sc3_std.append(np.std(t_sc3))
#     sc4_mean.append(np.mean(t_sc4))
#     sc4_std.append(np.std(t_sc4))
#     sc5_mean.append(np.mean(t_sc5))
#     sc5_std.append(np.std(t_sc5))
#
#     p1_mean.append(np.mean(t_p1))
#     p1_std.append(np.std(t_p1))
#     p2_mean.append(np.mean(t_p2))
#     p2_std.append(np.std(t_p2))
#     p3_mean.append(np.mean(t_p3))
#     p3_std.append(np.std(t_p3))
#     p4_mean.append(np.mean(t_p4))
#     p4_std.append(np.std(t_p4))
#     p5_mean.append(np.mean(t_p5))
#     p5_std.append(np.std(t_p5))
#
#     u1_mean.append(np.mean(t_u1))
#     u1_std.append(np.std(t_u1))
#     u2_mean.append(np.mean(t_u2))
#     u2_std.append(np.std(t_u2))
#     u3_mean.append(np.mean(t_u3))
#     u3_std.append(np.std(t_u3))
#     u4_mean.append(np.mean(t_u4))
#     u4_std.append(np.std(t_u4))
#     u5_mean.append(np.mean(t_u5))
#     u5_std.append(np.std(t_u5))
#
#     ratio1_mean.append(np.mean(r1))
#     ratio1_std.append(np.std(r1))
#     ratio2_mean.append(np.mean(r2))
#     ratio2_std.append(np.std(r2))
#     ratio3_mean.append(np.mean(r3))
#     ratio3_std.append(np.std(r3))
#     ratio4_mean.append(np.mean(r4))
#     ratio4_std.append(np.std(r4))
#     ratio5_mean.append(np.mean(r5))
#     ratio5_std.append(np.std(r5))
#
#
# def plot_bar_char1(y1_mean, y1_std, x_str, y_str, model_1):
#     plt.figure(figsize=(7, 6))
#     ind = np.arange(len(y1_mean)) * 2
#     label = range(min_x, max_x, step_x)
#     width = 0.8
#     plt.bar(ind, y1_mean, width=width, yerr=y1_std)
#     plt.xticks(ind, label)
#     plt.xlabel(x_str, fontsize=label_font_size)
#     plt.ylabel(y_str, fontsize=label_font_size)
#     plt.legend(model_1, fontsize=legend_font_size)
#     plt.show()
#
#
# def plot_bar_char2(y1_mean, y2_mean, y1_std, y2_std, x_str, y_str, model_1, model_2):
#     plt.figure(figsize=(7, 6))
#     ind = np.arange(len(y1_mean)) * 2
#     label = range(min_x, max_x, step_x)
#     width = 0.8
#     plt.bar(ind - width / 2, y1_mean, width=width, yerr=y1_std)
#     plt.bar(ind + width / 2, y2_mean, width=width, yerr=y2_std)
#     plt.xticks(ind, label)
#     plt.legend([model_1, model_2], fontsize=legend_font_size)
#     plt.xlabel(x_str, fontsize=label_font_size)
#     plt.ylabel(y_str, fontsize=label_font_size)
#     plt.show()
#
#
# def plot_bar_char3(y1_mean, y2_mean, y3_mean, y1_std, y2_std, y3_std, x_str, y_str, model_1, model_2, model_3):
#     plt.figure(figsize=(7, 6))
#     ind = np.arange(len(y1_mean)) * 2
#     width = 0.6
#     plt.bar(ind - width, y1_mean, width=width, yerr=y1_std)
#     plt.bar(ind, y2_mean, width=width, yerr=y2_std)
#     plt.bar(ind + width, y3_mean, width=width, yerr=y3_std)
#     plt.xticks(ind, range(min_x, max_x, step_x))
#     plt.legend([model_1, model_2, model_3], fontsize=legend_font_size)
#     plt.xlabel(x_str, fontsize=label_font_size)
#     plt.ylabel(y_str, fontsize=label_font_size)
#     plt.show()
#
#
# def plot_trend_char2(y1_mean, y2_mean, x_str, y_str, model_1, model_2):
#     plt.figure(figsize=(7, 6))
#     ind = range(len(y1_mean))
#     label = range(min_x, max_x, step_x)
#     plt.plot(ind, y1_mean, 'r-s', linewidth=3, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.plot(ind, y2_mean, 'float_value2-^', linewidth=3, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.xticks(ind, label)
#     plt.legend([model_1, model_2], fontsize=legend_font_size)
#     plt.grid(True)
#     plt.xlabel(x_str, fontsize=label_font_size)
#     plt.ylabel(y_str, fontsize=label_font_size)
#     plt.show()
#
#
# def plot_trend_char3(y1_mean, y2_mean, y3_mean, x_str, y_str, model_1, model_2, model_3):
#     plt.figure(figsize=(8, 6))
#     ind = range(len(y1_mean))
#     label = range(min_x, max_x, step_x)
#     plt.errorbar(ind, y1_mean, fmt='g-v', linewidth=4, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.errorbar(ind, y2_mean, fmt='r-s', linewidth=4, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.errorbar(ind, y3_mean, fmt='float_value2-^', linewidth=4, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.xticks(ind, label)
#     plt.legend([model_1, model_2, model_3], fontsize=legend_font_size)
#     plt.grid(True)
#     plt.xlabel(x_str, fontsize=label_font_size)
#     plt.ylabel(y_str, fontsize=label_font_size)
#     plt.show()
#
#
# def plot_trend_char4(y1_mean, y2_mean, y3_mean, y4_mean, x_str, y_str, model_1, model_2, model_3, model_4):
#     plt.figure(figsize=(7, 6))
#     ind = range(len(y1_mean))
#     label = range(min_x, max_x, step_x)
#     plt.errorbar(ind, y1_mean, fmt='g-v', linewidth=4.5, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.errorbar(ind, y2_mean, fmt='r-s', linewidth=4.5, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.errorbar(ind, y3_mean, fmt='float_value2-^', linewidth=4.5, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.errorbar(ind, y4_mean, fmt='y-o', linewidth=4.5, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.xticks(ind, label)
#     plt.legend([model_1, model_2, model_3, model_4], fontsize=legend_font_size)
#     plt.grid(True)
#     plt.xlabel(x_str, fontsize=label_font_size)
#     plt.ylabel(y_str, fontsize=label_font_size)
#     plt.show()
#
#
# def plot_trend_char5(y1_mean, y2_mean, y3_mean, y4_mean, y5_mean, x_str, y_str, model_1, model_2, model_3, model_4, model_5):
#     plt.figure(figsize=(8, 6))
#     ind = range(len(y1_mean))
#     label = range(min_x, max_x, step_x)
#     plt.errorbar(ind, y1_mean, fmt='y-o', linewidth=4, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.errorbar(ind, y2_mean, fmt='r-s', linewidth=4, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.errorbar(ind, y3_mean, fmt='float_value2-^', linewidth=4, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.errorbar(ind, y4_mean, fmt='g-v', linewidth=4, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.errorbar(ind, y5_mean, fmt='k-<', linewidth=4, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.xticks(ind, label)
#     plt.legend([model_1, model_2, model_3, model_4, model_5], fontsize=legend_font_size)
#     plt.grid(True)
#     plt.xlabel(x_str, fontsize=label_font_size)
#     plt.ylabel(y_str, fontsize=label_font_size)
#     plt.show()
#
#
# def plot_trend_error_char3(y1_mean, y2_mean, y3_mean,
#                            x_str, y_str,
#                            model_1, model_2, model_3,
#                            y_error1, y_error2, y_error3):
#     plt.figure(figsize=(8, 5))
#     ind = range(len(y1_mean))
#     label = range(min_x, max_x, step_x)
#     plt.errorbar(ind, y1_mean, yerr=y_error1, fmt='g-', linewidth=2, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.errorbar(ind, y2_mean, yerr=y_error2, fmt='r-', linewidth=2, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.errorbar(ind, y3_mean, yerr=y_error3, fmt='float_value2-', linewidth=2, markersize=10, markerfacecolor='w', markeredgewidth=3)
#     plt.tick_params(labelsize=15)  # 设置坐标轴字体
#     plt.xticks(ind, label)
#     plt.legend([model_1, model_2, model_3], fontsize=legend_font_size)
#     plt.grid(True)
#     plt.xlabel(x_str, fontsize=label_font_size)
#     plt.ylabel(y_str, fontsize=label_font_size)
#     plt.show()
#
#
# def plot_trend_error_char5(y1_mean, y2_mean, y3_mean, y4_mean, y5_mean,
#                            x_str, y_str,
#                            model_1, model_2, model_3, model_4, model_5,
#                            y_error1, y_error2, y_error3, y_error4, y_error5):
#     plt.figure(figsize=(8, 5))
#     ind = range(len(y1_mean))
#     label = range(min_x, max_x, step_x)
#     plt.errorbar(ind, y1_mean, yerr=y_error1, fmt='y-', linewidth=2, markersize=5, markerfacecolor='w', markeredgewidth=3)
#     plt.errorbar(ind, y2_mean, yerr=y_error2, fmt='r-', linewidth=2, markersize=5, markerfacecolor='w', markeredgewidth=3)
#     plt.errorbar(ind, y3_mean, yerr=y_error3, fmt='float_value2-', linewidth=2, markersize=5, markerfacecolor='w', markeredgewidth=3)
#     plt.errorbar(ind, y4_mean, yerr=y_error4, fmt='g-', linewidth=2, markersize=5, markerfacecolor='w', markeredgewidth=3)
#     plt.errorbar(ind, y5_mean, yerr=y_error5, fmt='k-', linewidth=2, markersize=5, markerfacecolor='w', markeredgewidth=3)
#     plt.tick_params(labelsize=15)   # 设置坐标轴字体
#     plt.xticks(ind, label)
#     plt.legend([model_1, model_2, model_3, model_4, model_5], fontsize=legend_font_size)
#     plt.grid(True)
#     plt.xlabel(x_str, fontsize=label_font_size)
#     plt.ylabel(y_str, fontsize=label_font_size)
#     plt.show()
#
# # 三个算法的对比
# plot_trend_char3(sw1_mean, sw2_mean, sw3_mean, "#Vehicles", "Social Welfare", model1, "MSWR-VCG",
#                  "MSWR-GM")
# plot_trend_char3(ratio1_mean, ratio2_mean, ratio3_mean, "#Vehicles",
#                  "Ratio of Served Orders", model1, "MSWR-VCG", "MSWR-GM")
#
# plot_trend_char3(sc1_mean, sc2_mean, sc3_mean, "#Vehicles", "Social Cost", model1, "MSWR-VCG", "MSWR-GM")
# plot_trend_char2(p2_mean, p3_mean, "#Vehicles", "Profit of the Platform", "MSWR-VCG", "MSWR-GM")
# plot_trend_char2(u2_mean, u3_mean, "#Vehicles", "Total Payoff of Drivers", "MSWR-VCG", "MSWR-GM")
# au1_mean = np.array(u1_mean) / np.arange(min_x, max_x, step_x)
# au1_std = np.array(u1_std) / np.arange(min_x, max_x, step_x)
# au2_mean = np.array(u2_mean) / np.arange(min_x, max_x, step_x)
# au2_std = np.array(u2_std) / np.arange(min_x, max_x, step_x)
# au3_mean = np.array(u3_mean) / np.arange(min_x, max_x, step_x)
# au3_std = np.array(u3_std) / np.arange(min_x, max_x, step_x)
# plot_trend_char2(au2_mean, au3_mean, "#Vehicles", "Average Payoff of Drivers", "MSWR-VCG", "MSWR-GM")
#
#
# # 四个算法的对比
# # plot_trend_char4(sw1_mean, sw2_mean, sw3_mean, sw4_mean, "Number of Vehicles", "Social Welfare", model1, model2,
# #                  model3, model4)
# # plot_trend_char4(ratio1_mean, ratio2_mean, ratio3_mean, ratio4_mean, "Number of Vehicles",
# #                  "Ratio of Served Orders", model1, model2, model3, model4)
# # plot_trend_char4(sc1_mean, sc2_mean, sc3_mean, sc4_mean, "Number of Vehicles", "Social Cost", model1, model2, model3, model4)
# # plot_trend_char3(p2_mean, p3_mean, p4_mean, "Number of Vehicles", "Profit of the Platform", model2, model3, model4)
# # plot_trend_char3(u2_mean, u3_mean, u4_mean, "Number of Vehicles", "Total Utilities of Drivers", model2, model3, model4)
# # au1_mean = np.array(u1_mean) / np.arange(min_x, max_x, step_x)
# # au1_std = np.array(u1_std) / np.arange(min_x, max_x, step_x)
# # au2_mean = np.array(u2_mean) / np.arange(min_x, max_x, step_x)
# # au2_std = np.array(u2_std) / np.arange(min_x, max_x, step_x)
# # au3_mean = np.array(u3_mean) / np.arange(min_x, max_x, step_x)
# # au3_std = np.array(u3_std) / np.arange(min_x, max_x, step_x)
# # au4_mean = np.array(u4_mean) / np.arange(min_x, max_x, step_x)
# # au4_std = np.array(u4_std) / np.arange(min_x, max_x, step_x)
# # plot_trend_char3(au2_mean, au3_mean, au4_mean, "Number of Vehicles", "Average Utilities of Drivers", model2, model3, model4, model5)
#
# # 五个算法的对比
# # plot_trend_char5(sw1_mean, sw2_mean, sw3_mean, sw4_mean, sw5_mean, "Number of Vehicles", "Social Welfare", model1, model2,
# #                  model3, model4, model5)
# # plot_trend_char5(ratio1_mean, ratio2_mean, ratio3_mean, ratio4_mean, ratio5_mean, "Number of Vehicles",
# #                  "Ratio of Served Orders", model1, model2, model3, model4, model5)
# # plot_trend_char5(sc1_mean, sc2_mean, sc3_mean, sc4_mean, sc5_mean, "Number of Vehicles", "Social Cost", model1, model2, model3, model4, model5)
# # plot_trend_char3(p4_mean, p2_mean, p3_mean, "Number of Vehicles", "Profit of the Platform", model4, model2, model3)
# # plot_trend_char3(u4_mean, u2_mean, u3_mean, "Number of Vehicles", "Total Utilities of Drivers", model4, model2, model3)
# # au1_mean = np.array(u1_mean) / np.arange(min_x, max_x, step_x)
# # au1_std = np.array(u1_std) / np.arange(min_x, max_x, step_x)
# # au2_mean = np.array(u2_mean) / np.arange(min_x, max_x, step_x)
# # au2_std = np.array(u2_std) / np.arange(min_x, max_x, step_x)
# # au3_mean = np.array(u3_mean) / np.arange(min_x, max_x, step_x)
# # au3_std = np.array(u3_std) / np.arange(min_x, max_x, step_x)
# # au4_mean = np.array(u4_mean) / np.arange(min_x, max_x, step_x)
# # au4_std = np.array(u4_std) / np.arange(min_x, max_x, step_x)
# # au5_mean = np.array(u5_mean) / np.arange(min_x, max_x, step_x)
# # au5_std = np.array(u5_std) / np.arange(min_x, max_x, step_x)
# # plot_trend_char3(au4_mean, au2_mean, au3_mean, "Number of Vehicles", "Average Utilities of Drivers", model4, model2, model3)
#
# # 五个算法误差对比
# # plot_trend_error_char5(sw1_mean, sw2_mean, sw3_mean, sw4_mean, sw5_mean,
# #                        "Number of Vehicles", "Social Welfare",
# #                        model1, model2, model3, model4, model5,
# #                        sw1_std, sw2_std, sw3_std, sw4_std, sw5_std)
# # plot_trend_error_char5(ratio1_mean, ratio2_mean, ratio3_mean, ratio4_mean, ratio5_mean,
# #                        "Number of Vehicles", "Ratio of Served Orders",
# #                        model1, model2, model3, model4, model5,
# #                        ratio1_std, ratio2_std, ratio3_std, ratio4_std, ratio5_std)
# # plot_trend_error_char5(sc1_mean, sc2_mean, sc3_mean, sc4_mean, sc5_mean, "Number of Vehicles", "Social Cost", model1, model2, model3, model4, model5,
# #                  sc1_std, sc2_std, sc3_std, sc4_std, sc5_std)
# # plot_trend_error_char3(p4_mean, p2_mean, p3_mean, "Number of Vehicles", "Profit of the Platform", model4, model2, model3,
# #                  p4_std, p2_std, p3_std)
# # plot_trend_error_char3(u4_mean, u2_mean, u3_mean, "Number of Vehicles", "Total Utilities of Drivers", model4, model2, model3,
# #                  u4_std, u2_std, u3_std)
# # au1_mean = np.array(u1_mean) / np.arange(min_x, max_x, step_x)
# # au1_std = np.array(u1_std) / np.arange(min_x, max_x, step_x)
# # au2_mean = np.array(u2_mean) / np.arange(min_x, max_x, step_x)
# # au2_std = np.array(u2_std) / np.arange(min_x, max_x, step_x)
# # au3_mean = np.array(u3_mean) / np.arange(min_x, max_x, step_x)
# # au3_std = np.array(u3_std) / np.arange(min_x, max_x, step_x)
# # au4_mean = np.array(u4_mean) / np.arange(min_x, max_x, step_x)
# # au4_std = np.array(u4_std) / np.arange(min_x, max_x, step_x)
# # au5_mean = np.array(u5_mean) / np.arange(min_x, max_x, step_x)
# # au5_std = np.array(u5_std) / np.arange(min_x, max_x, step_x)
# # plot_trend_error_char3(au4_mean, au2_mean, au3_mean, "Number of Vehicles", "Average Utilities of Drivers", model4, model2, model3,
# #                        au4_std, au2_std, au3_std)
