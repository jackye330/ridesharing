#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/2/19
import osmnx as ox

# graph = ox.graph_from_place("Manhattan, New York City, New York, USA", network_type="drive")
# graph_project = ox.project_graph(graph)
# ox.plot_graph(graph_project)
# ox.save_graphml(graph, filename="Manhattan.graphml", folder="./")
#
# place = ox.gdf_from_place("Manhattan, New York City, New York, USA")
# place = ox.project_gdf(place, to_latlong=True)  # 使用经纬度保存数据
# ox.save_gdf_shapefile(place, "Manhattan", folder="./")

graph = ox.graph_from_place("Chicago, USA", network_type="drive")
graph_project = ox.project_graph(graph)
ox.plot_graph(graph_project)
ox.save_graphml(graph, filename="Chicago.graphml", folder="./")

place = ox.gdf_from_place("Chicago, USA")
place = ox.project_gdf(place, to_latlong=True)  # 使用经纬度保存数据
ox.save_gdf_shapefile(place, "Chicago", folder="./")
ox.plot_shape(place)
