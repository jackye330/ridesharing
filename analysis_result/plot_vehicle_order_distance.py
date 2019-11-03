import pickle
from setting import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
#
# distance = []
# with open("./result/per_min/SWMOM-VCG/500_0.pkl", "rb") as f:
#     result = pickle.load(f)
#     time = result[8]
# sum = 0
# exceed_sum = 0
# for i in range(899, 1440):
#     for pair in time[i]:
#         sum += 1
#         if pair[0] < pair[1]:
#             exceed_sum += 1
# print(sum)
# print(exceed_sum)

with open("./result/per_min/SWMOM-VCG/500_0.pkl", "rb") as f:
    result = pickle.load(f)
    time = result[8]
sum = 0
print(result[9])

for t in range(899, 1440):
    for order in time[t]:
        pair = time[t][order]
        if len(pair) > 0:
            sum += 1
            print(order.order_id, pair)
print(sum)
