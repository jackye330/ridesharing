import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
min = 500
max = 2000
step = 100
model1 = "order_dispatch_one_simulation"
model1_ = "SPARP"
model2 = "orders_matching_with_gm"
model2_ = "SWMOM-GM"
model3 = "orders_matching_with_nearest_matching"
model3_ = "Nearest-Matching"
model4 = "orders_matching_with_vcg"
model4_ = "SWMOM-VCG"
model5 = "orders_matching_with_random"
model5_ = "Random"
for v in range(500, 2100, 100):
    with open("./result/running_time/{0}_{1}.pkl".format(model1, v), "rb") as file:
        result = pickle.load(file)
        scipy.io.savemat("./result_mat/{0}/{0}_{1}.mat".format(model1_, v), mdict={"res": result})
    with open("./result/running_time/{0}_{1}.pkl".format(model2, v), "rb") as file:
        result = pickle.load(file)
        scipy.io.savemat("./result_mat/{0}/{0}_{1}.mat".format(model2_, v), mdict={"res": result})
    with open("./result/running_time/{0}_{1}.pkl".format(model3, v), "rb") as file:
        result = pickle.load(file)
        scipy.io.savemat("./result_mat/{0}/{0}_{1}.mat".format(model3_, v), mdict={"res": result})
    with open("./result/running_time/{0}_{1}.pkl".format(model4, v), "rb") as file:
        result = pickle.load(file)
        scipy.io.savemat("./result_mat/{0}/{0}_{1}.mat".format(model4_, v), mdict={"res": result})
    with open("./result/running_time/{0}_{1}.pkl".format(model5, v), "rb") as file:
        result = pickle.load(file)
        scipy.io.savemat("./result_mat/{0}/{0}_{1}.mat".format(model5_, v), mdict={"res": result})
