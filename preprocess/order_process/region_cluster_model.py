#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/12/9
from typing import Dict
from collections import defaultdict
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans


class RegionModel:
    """
    这个类可以根据输入的类坐标位置，返回区域编号
    也可以根据区域编号返回这个区域内所有的坐标编号
    也可以随机返回这个区域内的一个随机的一个坐标号
    """
    __slots__ = ["index2region_id", "region_id2index", "region_number"]

    def __init__(self, index2region_id: Dict[int, int]):
        self.index2region_id = index2region_id
        self.region_id2index = defaultdict(list)
        for index, region_id in self.index2region_id.items():
            self.region_id2index[region_id].append(index)

        for region_id in self.region_id2index:
            self.region_id2index[region_id] = list(set(self.region_id2index[region_id]))

        self.region_number = len(self.region_id2index)

    def get_region_id_by_index(self, index: int):
        return self.index2region_id[index]

    def get_all_index_by_region_id(self, region_id: int):
        return self.region_id2index[region_id]

    def get_rand_index_by_region_id(self, region_id: int):
        return np.random.choice(self.region_id2index[region_id])


if __name__ == '__main__':
    n_cluster = 40
    df = pd.read_csv("../raw_data/points.csv")
    pick_cluster = MiniBatchKMeans(n_clusters=n_cluster, batch_size=100000).fit(df[["pick_lon", "pick_lat"]].values).predict(df[["pick_lon", "pick_lat"]])
    drop_cluster = MiniBatchKMeans(n_clusters=n_cluster, batch_size=100000).fit(df[["drop_lon", "drop_lat"]].values).predict(df[["drop_lon", "drop_lat"]])
    plt.scatter(
        x=df.pick_lon.values[:70000],
        y=df.pick_lat.values[:70000],
        c=pick_cluster[:70000],
        cmap="Paired",
        s=10
    )
    plt.title("Pick up Location Region")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig('./pick_up_location_region.eps', format='eps', dpi=1000)
    plt.figure()
    plt.scatter(
        x=df.drop_lon.values[:70000],
        y=df.drop_lat.values[:70000],
        c=drop_cluster[:70000],
        cmap="Paired",
        s=10
    )
    plt.title("Drop off Location Region")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("./drop_off_location_region.eps", format='eps', dpi=1000)
    # plt.show()

    pick_index2region_id = dict()
    drop_index2region_id = dict()
    pick_indexes = df.pick_index.values
    drop_indexes = df.drop_index.values

    for i in range(df.shape[0]):
        pick_index2region_id[pick_indexes[i]] = pick_cluster[i]
        drop_index2region_id[drop_indexes[i]] = drop_cluster[i]

    pick_region_model = RegionModel(pick_index2region_id)
    drop_region_model = RegionModel(drop_index2region_id)

    with open("../../data/Manhattan/order_data/order_model/pick_region_model.pkl", "wb") as f:
        pickle.dump(pick_region_model, f)

    with open("../../data/Manhattan/order_data/order_model/drop_region_model.pkl", "wb") as f:
        pickle.dump(drop_region_model, f)

    print(pick_region_model.region_number)
    print(drop_region_model.region_number)
