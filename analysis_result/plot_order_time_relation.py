import pickle
from setting import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

order = []
order_sum = 0
with open("./env_data/order/500_0.pkl", "rb") as f:
    result = pickle.load(f)
for i in range(0, 1440):
    order_sum += len(result[i])
    order.append(len(result[i]))
print(order_sum)
plt.plot(order)
plt.show()

