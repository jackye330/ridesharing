#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/11/3
"""
用于统一接口
"""
from setting import EXPERIMENTAL_MODE

if EXPERIMENTAL_MODE == "real":
    from .transport_env import _Graph
else:
    from .grid_env import _Graph

Network = _Graph
