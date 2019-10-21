import numpy as np
import time
import pickle
import scipy.io
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#
# d = {}
# d['fun'] = 1
# d['fuck'] = 2
# d['fff'] = 3
# ll = sorted(d.items(), key=lambda d: d[1], reverse=True)
# print(ll[0][1])
# print(np.power(2, 3))
cur_list = []
rem_list = [5, 8]

def trackback(cur_list, rem_list):
    print("cur_list", cur_list)
    print("rem_list", rem_list)
    if len(cur_list) == 4:
        return
    for p in rem_list:
        cur_list_ = cur_list.copy()
        cur_list_.append(p)
        rem_list_ = rem_list
        if p == 5 and not rem_list_.__contains__(6):
            rem_list_.append(6)
        if p == 8 and not rem_list_.__contains__(14):
            rem_list_.append(14)
        trackback(cur_list_, rem_list_)


# trackback(cur_list, rem_list)
# start = time.clock()
# a = [1, 2, 3]
# b = []
# for i in range(3):
#     b.append(a[i])
# b[0] = 4
# print(a)
# print(b)
# elapsed = time.clock() - start
# print("time used", elapsed, "s")

# with open("./env_data/static.pkl", "rb") as file:
#     static = pickle.load(file)
#     graph, shortest_distance, shortest_path, shortest_path_with_minute = static
# print(shortest_distance[2641][936])
# print(shortest_distance[936][3293])
# print(shortest_distance[3293][2624])
# print(shortest_distance[2624][98])
# print(shortest_distance[3293][98])
# print(shortest_distance[2641][3293])
# with open("./result/{0}/{1}_{2}.pkl".format("Random", 500, 0), "rb") as file:
#     result1 = pickle.load(file)
#     social_welfare_result1, social_cost_result1, _, total_utility_result1, total_profit_result1, service_rate1 = result1
# print(total_utility_result1)
# print(social_welfare_result1)
# print(social_cost_result1)
# print(total_profit_result1)
# print(service_rate1)
#         model1 = "Nearest-Matching"
#         v=2000
#         k=1
#         with open("./result/{0}/{1}_{2}.pkl".format(model1, v, k), "rb") as file:
#             result1 = pickle.load(file)
#             social_welfare_result1, social_cost_result1, _, total_utility_result1, total_profit_result1, service_rate1 = result1
# with open("./result/{0}/{1}_{2}.pkl".format("Random", 1000, 1), "rb") as file:
#             result1 = pickle.load(file)
#             social_welfare_result, social_cost_result, _, total_utility_result, total_profit_result, service_rate = result1
# # print(social_welfare_result)
# with open("./env_data/order/1500_0.pkl", "rb") as f:
#     r = pickle.load(f)
# for i in range(799, 1439):
#     print(len(r[i]))
# 为什么最后匹配率都很低？
# time = []
# time1 = []
# model = "SWMOM-GM"
# model1 = "Nearest-Matching"
# for i in range(0, 1):
#     with open("./result/per_min/{0}/{1}_{2}.pkl".format(model, 500, i), "rb") as f:
#         result = pickle.load(f)
#         time.append(result[5])
#     with open("./result/per_min/{0}/{1}_{2}.pkl".format(model1, 500, i), "rb") as f:
#         result1 = pickle.load(f)
#         time1.append(result1[5])
# time = np.array(time)
# time = time.mean(axis=0)
# time1 = np.array(time1)
# time1 = time1.mean(axis=0)
# legend_fontsize = 14
# label_fontsize = 16
# plt.figure(figsize=(7, 6))
# ind = np.arange(len(time1))
# width = 0.8
# plt.plot(ind, time1, time)
# plt.xlabel("time", fontsize=label_fontsize)
# plt.ylabel("service rate", fontsize=label_fontsize)
# plt.legend(["Nearest-Matching", "SWMOM-GM"], fontsize=legend_fontsize)  # "SWMOM-GM",
# plt.show()
# 每分钟未匹配订单情况，每分钟匹配订单情况
unmatched = []
matched = []
sw = []
service_rate = []
empty_vehicle_rate = []
model = "Nearest-Matching"
run_type_part = "700_1440"
run_type_dynamic = "dynamic_500"
for i in range(0, 1):
    with open("./result/per_min/{0}/{1}_{2}.pkl".format(model, 500, i), "rb") as f:
        result = pickle.load(f)
        unmatched.append(result[6])
        matched.append(result[7])
        sw.append(result[0])
        service_rate.append(result[5])
        empty_vehicle_rate.append((result[8]))
print(result[9])
legend_fontsize = 14
label_fontsize = 16
plt.figure(figsize=(7, 6))
ind = np.arange(len(unmatched[0]))
width = 0.8
plt.plot(ind, unmatched[0])
plt.xlabel("time", fontsize=label_fontsize)
plt.ylabel("unmatched_order_per_min", fontsize=label_fontsize)
plt.legend([model], fontsize=legend_fontsize)  # "SWMOM-GM",
plt.show()
plt.plot(ind, matched[0])
plt.xlabel("time", fontsize=label_fontsize)
plt.ylabel("matched_order_per_min", fontsize=label_fontsize)
plt.legend([model], fontsize=legend_fontsize)
plt.show()
plt.plot(ind, service_rate[0])
plt.xlabel("time", fontsize=label_fontsize)
plt.ylabel("service rate", fontsize=label_fontsize)
plt.legend([model], fontsize=legend_fontsize)  # "SWMOM-GM",
plt.show()
plt.plot(ind, sw[0])
plt.xlabel("time", fontsize=label_fontsize)
plt.ylabel("social welfare", fontsize=label_fontsize)
plt.legend([model], fontsize=legend_fontsize)  # "SWMOM-GM",
plt.show()
plt.plot(ind, empty_vehicle_rate[0])
plt.xlabel("time", fontsize=label_fontsize)
plt.ylabel("empty_vehicle_rate", fontsize=label_fontsize)
plt.legend([model], fontsize=legend_fontsize)  # "SWMOM-GM",
plt.show()


