import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

plt.rc('font', family='Times New Roman', weight=3)
model1 = "Nearest-Matching"
model2 = "SWMOM-VCG"
model3 = "SWMOM-GM"
model4 = "SPARP"
model5 = "Random"

min_x = 500
max_x = 2501
step_x = 500
n_sample = 10
legend_fontsize = 14
label_fontsize = 16


sw1_mean, sw1_std = [], []
sw2_mean, sw2_std = [], []
sw3_mean, sw3_std = [], []
sw4_mean, sw4_std = [], []
sw5_mean, sw5_std = [], []
sc1_mean, sc1_std = [], []
sc2_mean, sc2_std = [], []
sc3_mean, sc3_std = [], []
sc4_mean, sc4_std = [], []
sc5_mean, sc5_std = [], []
u1_mean, u1_std = [], []
u2_mean, u2_std = [], []
u3_mean, u3_std = [], []
u4_mean, u4_std = [], []
u5_mean, u5_std = [], []
p1_mean, p1_std = [], []
p2_mean, p2_std = [], []
p3_mean, p3_std = [], []
p4_mean, p4_std = [], []
p5_mean, p5_std = [], []
ratio1_mean, ratio1_std = [], []
ratio2_mean, ratio2_std = [], []
ratio3_mean, ratio3_std = [], []
ratio4_mean, ratio4_std = [], []
ratio5_mean, ratio5_std = [], []
for v in range(min_x, max_x, step_x):
    t_sw1 = []
    t_sw2 = []
    t_sw3 = []
    t_sw4 = []
    t_sw5 = []
    t_sc1 = []
    t_sc2 = []
    t_sc3 = []
    t_sc4 = []
    t_sc5 = []
    t_p1 = []
    t_p2 = []
    t_p3 = []
    t_p4 = []
    t_p5 = []
    t_u1 = []
    t_u2 = []
    t_u3 = []
    t_u4 = []
    t_u5 = []
    r1 = []
    r2 = []
    r3 = []
    r4 = []
    r5 = []
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
        with open("./result/{0}/{1}_{2}.pkl".format(model4, v, k), "rb") as file:
            result4 = pickle.load(file)
            social_welfare_result4, social_cost_result4, _, total_utility_result4, total_profit_result4, service_rate4 = result4
        with open("./result/{0}/{1}_{2}.pkl".format(model5, v, k), "rb") as file:
            result5 = pickle.load(file)
            social_welfare_result5, social_cost_result5, _, total_utility_result5, total_profit_result5, service_rate5 = result5
        t_sw1.append(np.sum(social_welfare_result1))
        t_sw2.append(np.sum(social_welfare_result2))
        t_sw3.append(np.sum(social_welfare_result3))
        t_sw4.append(np.sum(social_welfare_result4))
        t_sw4.append(np.sum(social_welfare_result4))
        t_sw5.append(np.sum(social_welfare_result5))
        t_sc1.append(np.sum(social_cost_result1))
        t_sc2.append(np.sum(social_cost_result2))
        t_sc3.append(np.sum(social_cost_result3))
        t_sc4.append(np.sum(social_cost_result4))
        t_sc5.append(np.sum(social_cost_result5))
        r1.append(service_rate1)
        r2.append(service_rate2)
        r3.append(service_rate3)
        r4.append(service_rate4)
        r5.append(service_rate5)
        t_p1.append(np.sum(total_profit_result1))
        t_p2.append(np.sum(total_profit_result2))
        t_p3.append(np.sum(total_profit_result3))
        t_p4.append(np.sum(total_profit_result4))
        t_p5.append(np.sum(total_profit_result5))
        t_u1.append(np.sum(total_utility_result1))
        t_u2.append(np.sum(total_utility_result2))
        t_u3.append(np.sum(total_utility_result3))
        t_u4.append(np.sum(total_utility_result4))
        t_u5.append(np.sum(total_utility_result5))

    sw1_mean.append(np.mean(t_sw1))
    sw1_std.append(np.std(t_sw1))
    sw2_mean.append(np.mean(t_sw2))
    sw2_std.append(np.std(t_sw2))
    sw3_mean.append(np.mean(t_sw3))
    sw3_std.append(np.std(t_sw3))
    sw4_mean.append(np.mean(t_sw4))
    sw4_std.append(np.std(t_sw4))
    sw5_mean.append(np.mean(t_sw5))
    sw5_std.append(np.std(t_sw5))

    sc1_mean.append(np.mean(t_sc1))
    sc1_std.append(np.std(t_sc1))
    sc2_mean.append(np.mean(t_sc2))
    sc2_std.append(np.std(t_sc2))
    sc3_mean.append(np.mean(t_sc3))
    sc3_std.append(np.std(t_sc3))
    sc4_mean.append(np.mean(t_sc4))
    sc4_std.append(np.std(t_sc4))
    sc5_mean.append(np.mean(t_sc5))
    sc5_std.append(np.std(t_sc5))

    p1_mean.append(np.mean(t_p1))
    p1_std.append(np.std(t_p1))
    p2_mean.append(np.mean(t_p2))
    p2_std.append(np.std(t_p2))
    p3_mean.append(np.mean(t_p3))
    p3_std.append(np.std(t_p3))
    p4_mean.append(np.mean(t_p4))
    p4_std.append(np.std(t_p4))
    p5_mean.append(np.mean(t_p5))
    p5_std.append(np.std(t_p5))

    u1_mean.append(np.mean(t_u1))
    u1_std.append(np.std(t_u1))
    u2_mean.append(np.mean(t_u2))
    u2_std.append(np.std(t_u2))
    u3_mean.append(np.mean(t_u3))
    u3_std.append(np.std(t_u3))
    u4_mean.append(np.mean(t_u4))
    u4_std.append(np.std(t_u4))
    u5_mean.append(np.mean(t_u5))
    u5_std.append(np.std(t_u5))

    ratio1_mean.append(np.mean(r1))
    ratio1_std.append(np.std(r1))
    ratio2_mean.append(np.mean(r2))
    ratio2_std.append(np.std(r2))
    ratio3_mean.append(np.mean(r3))
    ratio3_std.append(np.std(r3))
    ratio4_mean.append(np.mean(r4))
    ratio4_std.append(np.std(r4))
    ratio5_mean.append(np.mean(r5))
    ratio5_std.append(np.std(r5))
au1_mean = np.array(u1_mean) / np.arange(min_x, max_x, step_x)
au1_std = np.array(u1_std) / np.arange(min_x, max_x, step_x)
au2_mean = np.array(u2_mean) / np.arange(min_x, max_x, step_x)
au2_std = np.array(u2_std) / np.arange(min_x, max_x, step_x)
au3_mean = np.array(u3_mean) / np.arange(min_x, max_x, step_x)
au3_std = np.array(u3_std) / np.arange(min_x, max_x, step_x)
au4_mean = np.array(u4_mean) / np.arange(min_x, max_x, step_x)
au4_std = np.array(u4_std) / np.arange(min_x, max_x, step_x)
au5_mean = np.array(u5_mean) / np.arange(min_x, max_x, step_x)
au5_std = np.array(u5_std) / np.arange(min_x, max_x, step_x)

print(sw5_mean)

scipy.io.savemat("./result_mat/social_welfare.mat", mdict={"random_mean": sw5_mean, "random_std": sw5_std,
                                                           "Nearest-Matching_mean": sw1_mean, "Nearest-Matching_std": sw1_std,
                                                           "SWMOM-VCG_mean": sw2_mean, "SWMOM-VCG_std": sw2_std,
                                                           "SWMOM-GM_mean": sw3_mean, "SWMOM-GM_std": sw3_std,
                                                           "SPARP_mean": sw4_mean, "SPARP_std": sw4_std})
scipy.io.savemat("./result_mat/social_cost.mat", mdict={"random_mean": sc5_mean, "random_std": sc5_std,
                                                           "Nearest-Matching_mean": sc1_mean, "Nearest-Matching_std": sc1_std,
                                                           "SWMOM-VCG_mean": sc2_mean, "SWMOM-VCG_std": sc2_std,
                                                           "SWMOM-GM_mean": sc3_mean, "SWMOM-GM_std": sc3_std,
                                                           "SPARP_mean": sc4_mean, "SPARP_std": sc4_std})
scipy.io.savemat("./result_mat/total_utility.mat", mdict={"random_mean": u5_mean, "random_std": u5_std,
                                                           "Nearest-Matching_mean": u1_mean, "Nearest-Matching_std": u1_std,
                                                           "SWMOM-VCG_mean": u2_mean, "SWMOM-VCG_std": u2_std,
                                                           "SWMOM-GM_mean": u3_mean, "SWMOM-GM_std": u3_std,
                                                           "SPARP_mean": u4_mean, "SPARP_std": u4_std})
scipy.io.savemat("./result_mat/total_profit.mat", mdict={"random_mean": p5_mean, "random_std": p5_std,
                                                           "Nearest-Matching_mean": p1_mean, "Nearest-Matching_std": p1_std,
                                                           "SWMOM-VCG_mean": p2_mean, "SWMOM-VCG_std": p2_std,
                                                           "SWMOM-GM_mean": p3_mean, "SWMOM-GM_std": p3_std,
                                                           "SPARP_mean": p4_mean, "SPARP_std": p4_std})
scipy.io.savemat("./result_mat/service_rate.mat", mdict={"random_mean": ratio5_mean, "random_std": ratio5_std,
                                                           "Nearest-Matching_mean": ratio1_mean, "Nearest-Matching_std": ratio1_std,
                                                           "SWMOM-VCG_mean": ratio2_mean, "SWMOM-VCG_std": ratio2_std,
                                                           "SWMOM-GM_mean": ratio3_mean, "SWMOM-GM_std": ratio3_std,
                                                           "SPARP_mean": ratio4_mean, "SPARP_std": ratio4_std})


