#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2019/1/23
import pandas as pd
# car_type 车辆类型
# product_company 生产企业
# driving_mode 驱动型式
# transmission 变速器类型
# engine_style 发动机类型
# power 额定功率
# weight 车辆净重量
# max_weight 车辆最大重量
# Suburban_fuel_consumption 市郊油耗
# Urban_fuel_consumption 市区油耗
# Comprehensive_fuel_consumption 综合油耗


raw_data = pd.read_csv("car_fuel_consumption_info.csv", error_bad_lines=False)
print(raw_data.shape)
export_data = raw_data[raw_data["车辆型号"] != "NULL"]
export_data = export_data[export_data["生产企业"] != "NULL"]
export_data = export_data[export_data["驱动型式"] != "NULL"]
export_data = export_data[export_data["变速器类型"] != "NULL"]
export_data = export_data[export_data["发动机型号"] != "NULL"]
export_data = export_data[export_data["额定功率"] != 0]
export_data = export_data[export_data["车辆净重量"] >= 870]
export_data = export_data[export_data["车辆最大重量"] >= 870]
export_data = export_data[export_data["市郊油耗"] != 0]
export_data = export_data[export_data["市区油耗"] != 0]
export_data = export_data[export_data["综合油耗"] != 0]

print(export_data.shape)
export_data.to_csv("new_car_fuel_consumption_info.csv", index=False)
