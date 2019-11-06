import pickle
from setting import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

with open('./result/SWMOM-GM/2000_0.pkl', "rb") as file:
    result = pickle.load(file)
print(result)
