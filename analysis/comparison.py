import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

model1="SPARP"
model2="SWMOM-VCG"

v = 500
t_sw1 = []
t_sw2 = []
t_sw3 = []
t_sw4 = []
t_sc1 = []
t_sc2 = []
t_sc3 = []
t_sc4 = []
t_p1 = []
t_p2 = []
t_p3 = []
t_p4 = []
t_u1 = []
t_u2 = []
t_u3 = []
t_u4 = []
r1 = []
r2 = []
r3 = []
r4 = []
for k in range(10):
    with open("./result/{0}/{1}_{2}.pkl".format(model1, v, k), "rb") as file:
        result1 = pickle.load(file)
        social_welfare_result1, social_cost_result1, _, total_utility_result1, total_profit_result1, service_rate1 = result1
    with open("./result/{0}/{1}_{2}.pkl".format(model2, v, k), "rb") as file:
        result2 = pickle.load(file)
        social_welfare_result2, social_cost_result2, _, total_utility_result2, total_profit_result2, service_rate2 = result2
    t_sw1.append(np.sum(social_welfare_result1))
    t_sw2.append(np.sum(social_welfare_result2))
    t_sc1.append(np.sum(social_cost_result1))
    t_sc2.append(np.sum(social_cost_result2))
    r1.append(service_rate1)
    r2.append(service_rate2)
    t_p1.append(np.sum(total_profit_result1))
    t_p2.append(np.sum(total_profit_result2))
    t_u1.append(np.sum(total_utility_result1))
    t_u2.append(np.sum(total_utility_result2))

print(t_u1)
print(t_u2)

plt.plot(t_sw1)
plt.plot(t_sw2)
plt.ylabel("social_welfare", fontsize=14)
plt.legend([model1, model2], fontsize=14)
plt.show()

plt.plot(t_sc1)
plt.plot(t_sc2)
plt.ylabel("social_cost", fontsize=14)
plt.legend([model1, model2], fontsize=14)
plt.show()

plt.plot(r1)
plt.plot(r2)
plt.ylabel("service_rate", fontsize=14)
plt.legend([model1, model2], fontsize=14)
plt.show()

plt.plot(t_p1)
plt.plot(t_p2)
plt.ylabel("platform_profit", fontsize=14)
plt.legend([model1, model2], fontsize=14)
plt.show()

plt.plot(t_u1)
plt.plot(t_u2)
plt.ylabel("total_driver_payoffs", fontsize=14)
plt.legend([model1, model2], fontsize=14)
plt.show()
