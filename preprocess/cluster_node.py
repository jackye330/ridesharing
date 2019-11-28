#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/27
import osmnx as ox
from typing import Set
from collections import defaultdict
from utility import is_enough_small


class Edge:
    __slots__ = ["p1", "p2", "distance", "oneway"]

    def __init__(self, p1_index: int, p2_index: int, distance: float, oneway: bool):
        self.p1 = p1_index
        self.p2 = p2_index
        self.distance = distance
        self.oneway = oneway

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise Exception("类型不一致")
        else:
            return self.p1 == other.p1 and self.p2 == other.p2

    def __hash__(self):
        return hash((self.p1, self.p2))

    def __lt__(self, other):
        return self.distance < other.distance


def merge_point(p1: int, p2: int):
    covered_points.add(p1)
    covered_points.add(p2)
    point_children[p1].add(p2)
    point_parent[p2] = p1


def check_merge_point_cluster(p1: int, p2: int):
    for p3 in point_children[p2]:
        if not is_enough_small(weights[p1, p2] + weights[p2, p3], threshold_distance):
            return False
    return True


def merge_point_cluster(p1: int, p2: int):
    """
    将以p2为中心的点集移动到p1的点上
    :param p1:
    :param p2:
    :return:
    """
    for p3 in point_children[p2]:
        point_children[p1].add(p3)
        point_parent[p3] = p1
        if (p1, p3) not in weights:
            weights[p1, p3] = weights[p1, p2] + weights[p2, p3]
        else:
            if not is_enough_small(weights[p1, p3], weights[p1, p2] + weights[p2, p3]):
                weights[p1, p3] = weights[p1, p2] + weights[p2, p3]
    point_children[p1].add(p2)
    point_parent[p2] = p1
    point_children.pop(p2)  # 以后不存在p2了


graph = ox.load_graphml("NewYork.graphml", "../data/NewYork/network_data/")
osm_id2index = {osm_id: i for i, osm_id in enumerate(graph.nodes)}
index2osm_id = {i: osm_id for i, osm_id in enumerate(graph.nodes)}
print(len(osm_id2index))

point_parent: dict = dict()  # 节点的父节点，如果是单节点那么他的父节点就是自己
point_children: defaultdict = defaultdict(set)  # 第i个节点的孩子
covered_points: Set[int] = set()  # 已经被处理的节点

edges = [Edge(osm_id2index[edge[0]], osm_id2index[edge[1]], edge[2]["length"], edge[2]["oneway"]) for edge in graph.edges(data=True)]
weights = {}

for edge in edges:
    weights[edge.p1, edge.p2] = edge.distance
    if not edge.oneway:
        weights[edge.p2, edge.p1] = edge.distance

edges.sort()
threshold_distance = 500.0
cnt = 0  # 已经处理了的边

for edge in edges:
    if edge.distance > threshold_distance:
        break
    cnt += 1
    if edge.p1 == edge.p2:
        continue
    if edge.p1 not in covered_points and edge.p2 not in covered_points:
        merge_point(edge.p1, edge.p2)
    elif edge.p1 in point_children and edge.p2 in point_children:
        if edge.oneway:
            if check_merge_point_cluster(edge.p1, edge.p2):
                merge_point_cluster(edge.p1, edge.p2)
        else:
            if check_merge_point_cluster(edge.p1, edge.p2):
                merge_point_cluster(edge.p1, edge.p2)
            elif check_merge_point_cluster(edge.p2, edge.p1):
                merge_point_cluster(edge.p2, edge.p1)
    elif edge.p1 in point_children and edge.p2 not in covered_points:
        merge_point(edge.p1, edge.p2)
    elif edge.p2 in point_children and edge.p1 not in covered_points:
        if not edge.oneway:  # 只有双向的才会有p2 -> p1
            merge_point(edge.p2, edge.p1)

center_points = set()
for point in range(len(osm_id2index)):
    if point in point_children:
        center_points.add(point)
    elif point not in covered_points:
        center_points.add(point)

print(len(center_points))

