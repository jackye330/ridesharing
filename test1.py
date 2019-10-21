import pickle
import numpy as np
from setting import *
import copy
from agent import *
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# node = 500
# average_speed = AVERAGE_SPEED
# with open("./env_data/static.pkl", "rb") as file:
#     static = pickle.load(file)
#     graph, shortest_distance, shortest_path, shortest_path_with_minute = static
# index = shortest_path_with_minute[node, :]
# print(type(shortest_path_with_minute))
# new_index = list(set(index))
# print(new_index)
# next_index = np.random.choice(shortest_path_with_minute[node])
# print(next_index)
# adjacent_nodes = np.load("./network_data/adjacent_nodes.npy")
# print(list(set(adjacent_nodes[node])))
# with open("./env_data/static.pkl", "rb") as file:
#     static = pickle.load(file)
#     graph, shortest_distance, shortest_path, shortest_path_with_minute, adjacent_nodes = static
# print(list(set(adjacent_nodes[2494])))
# print(list(set(shortest_path_with_minute[4321])))
# print(shortest_distance[4321, 1918])
# print(shortest_path[4321, 1918])
# print(shortest_distance[4321, 120])
# with open("./env_data/order/{0}_{1}.pkl".format(3000, 0), "rb") as file:
#     orders = pickle.load(file)
# # 复制一遍订单，变成两天的数据
# new_orders = copy.deepcopy(orders)
# print(len(orders[1310]))
# for t in range(MIN_REQUEST_TIME, MAX_REQUEST_TIME):
#     for order in new_orders[t]:
#         order.request_time += 1440
#     orders[t + 1440] = new_orders[t]
# print(len(orders[2750]))
vehicle_num = 2000
repeat = 0
sw1 = []
sw2 = []
sw3 = []
sw4 = []
sc1 = []
sc2 = []
sc3 = []
sc4 = []
pp1 = []
pp2 = []
pp3 = []
pp4 = []
pd1 = []
pd2 = []
pd3 = []
pd4 = []
sw1_mean = []
sw2_mean = []
sw3_mean = []
sw4_mean = []
sc1_mean = []
sc2_mean = []
sc3_mean = []
sc4_mean = []
pp1_mean = []
pp2_mean = []
pp3_mean = []
pp4_mean = []
pd1_mean = []
pd2_mean = []
pd3_mean = []
pd4_mean = []
with open("./result/per_min/{0}/{1}_{2}.pkl".format("SWMOM-VCG", vehicle_num, repeat), "rb") as f:
    result = pickle.load(f)
    sw1.append(result[0])
    sc1.append(result[1])
    pp1.append(result[4])
    pd1.append(result[3])
with open("./result/per_min/{0}/{1}_{2}.pkl".format("SWMOM-GM", vehicle_num, repeat), "rb") as f:
    result = pickle.load(f)
    sw2.append(result[0])
    sc2.append(result[1])
    pp2.append(result[4])
    pd2.append(result[3])
with open("./result/per_min/{0}/{1}_{2}.pkl".format("SPARP", vehicle_num, repeat), "rb") as f:
    result = pickle.load(f)
    sw3.append(result[0])
    sc3.append(result[1])
    pp3.append(result[4])
    pd3.append(result[3])
with open("./result/per_min/{0}/{1}_{2}.pkl".format("Nearest-Matching", vehicle_num, repeat), "rb") as f:
    result = pickle.load(f)
    sw4.append(result[0])
    sc4.append(result[1])
    pp4.append(result[4])
    pd4.append(result[3])
sw1_mean.append(np.mean(sw1[0]))
sw2_mean.append(np.mean(sw2[0]))
sw3_mean.append(np.mean(sw3[0]))
sw4_mean.append(np.mean(sw4[0]))
sc1_mean.append(np.mean(sc1[0]))
sc2_mean.append(np.mean(sc2[0]))
sc3_mean.append(np.mean(sc3[0]))
sc4_mean.append(np.mean(sc4[0]))
pp1_mean.append(np.mean(pp1[0]))
pp2_mean.append(np.mean(pp2[0]))
pp3_mean.append(np.mean(pp3[0]))
pp4_mean.append(np.mean(pp4[0]))
pd1_mean.append(np.mean(pd1[0]))
pd2_mean.append(np.mean(pd2[0]))
pd3_mean.append(np.mean(pd3[0]))
pd4_mean.append(np.mean(pd4[0]))
print("social welfare")
print(sw1_mean)
print(sw2_mean)
print(sw3_mean)
print(sw4_mean)
print("social cost")
print(sc1_mean)
print(sc2_mean)
print(sc3_mean)
print(sc4_mean)
print("profit of platform")
print(pp1_mean)
print(pp2_mean)
print(pp3_mean)
print(pp4_mean)
print("profits of drivers")
print(pd1_mean)
print(pd2_mean)
print(pd3_mean)
print(pd4_mean)
plt.plot(sw1_mean[0])
plt.plot(sw2_mean[0])
plt.plot(sw3_mean[0])
plt.plot(sw4_mean[0])
plt.legend(["SWMOM-VCG", "SWMOM-GM", "SPARP", "Nearest-Matching"])
plt.show()
