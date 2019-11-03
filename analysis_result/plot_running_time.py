# date   : 2019/3/21
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Times New Roman', weight=3)
model_name = ["Nearest-Matching", "MSWR-VCG", "MSWR-GM"]
style = ['g-v', 'r-s', 'b-^']
model1 = "orders_matching_with_nearest_matching"
model2 = "orders_matching_with_vcg"
model3 = "orders_matching_with_gm"
model4 = "order_dispatch_one_simulation"
min_x, max_x, step_x = 500, 2501, 500
label = range(min_x, max_x, step_x)
legend_fontsize = 14
label_fontsize = 16
plt.figure(figsize=(7, 6))
data_sum1 = []
data_sum2 = []
data_sum3 = []
for v in range(500, 2501, 500):
    with open("./result/running_time/{0}_{1}.pkl".format(model1, v), "rb") as file:
        data1 = pickle.load(file)
    with open("./result/running_time/{0}_{1}.pkl".format(model2, v), "rb") as file:
        data2 = pickle.load(file)
    with open("./result/running_time/{0}_{1}.pkl".format(model3, v), "rb") as file:
        data3 = pickle.load(file)
    print(len(data1))
    print(len(data2))
    print(len(data3))
    print(" ")
    data1 = np.array(data1)
    data_sum1.append(np.sum(data1, axis=1))
    data2 = np.array(data2)
    data_sum2.append(np.sum(data2, axis=1))
    data3 = np.array(data3)
    data_sum3.append(np.sum(data3, axis=1))
data_sum1 = np.array(data_sum1)
# data_sum1 = data_sum1.mean(axis=1)
data_sum2 = np.array(data_sum2)
# data_sum2 = data_sum2.mean(axis=1)
data_sum3 = np.array(data_sum3)
# data_sum3 = data_sum3.mean(axis=1)
# print(data_sum1)
# print(data_sum2)
# print(data_sum3)
# ind = range(len(data_sum1))
# plt.plot(ind, data_sum1, style[0], linewidth=3, markersize=10, markerfacecolor='w', markeredgewidth=3)
# plt.plot(ind, data_sum2, style[0], linewidth=3, markersize=10, markerfacecolor='w', markeredgewidth=3)
# plt.plot(ind, data_sum2, style[0], linewidth=3, markersize=10, markerfacecolor='w', markeredgewidth=3)
# plt.xticks(ind, label)
# plt.legend(model_name, fontsize=legend_fontsize)
# plt.grid(True)
# plt.xlabel("#Vehicles", fontsize=label_fontsize)
# plt.ylabel("Running Time (s)", fontsize=label_fontsize)
# plt.show()
