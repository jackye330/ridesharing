#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/5/30
import osmnx as ox
import matplotlib.pyplot as plt
graph = ox.load_graphml(filename="Manhattan.graphml", folder="./")
# plt.figure(figsize=(5,5))
ox.plot_graph(graph)