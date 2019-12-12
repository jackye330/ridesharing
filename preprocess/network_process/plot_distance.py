#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/12/9
import numpy as np
import matplotlib.pyplot as plt
shortest_distance = np.load("../../data/Manhattan/network_data/shortest_distance.npy")
d = shortest_distance.flatten()
d = d[np.where(d > 1)]
d = d[np.where(d != np.inf)]
plt.hist(d, bins=100)
plt.show()
