# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:10:09 2024

@author: AC512
"""

import geopandas as gpd
from shapely.geometry import LineString
import os
import pandas as pd

# 定義路徑
root = r'D:\Users\ae133\Desktop\花蓮智慧國土\電子地圖中線\output'
point_path = os.path.join(root, 'combined_output_new.shp')

# 儲存所有生成的線段
line_output = []

# 讀取點資料
gdf = gpd.read_file(point_path)

# 確認資料的前幾行和屬性欄位
print("點資料及屬性:")
print(gdf.head())  # 顯示前5行

# 根據 ROADSEGID 分組
grouped = gdf.groupby('ROADSEGID')

# 遍歷每個群組並儲存
for roadsegid, group in grouped:
    # 依照 P_number 進行排序
    group = group.sort_values(by='p_number')
    
    # 確保群組中至少有兩個點才能生成線段
    if len(group) > 1:
        # 生成線段，確保經過所有排序後的中間點
        line = LineString(group.geometry.tolist())

        # 繼承第一個點的屬性
        first_point_attributes = group.iloc[0].drop(['geometry', 'p_number'])  # 取得第一個點的屬性，排除 geometry 和 p_number
        
        # 將 None 替換為空字串
        first_point_attributes = first_point_attributes.fillna("")  # 將 None 替換為空白字串
        
        # 將線段的屬性轉換為 DataFrame
        line_attributes = first_point_attributes.to_frame().T  # 將 Series 轉換為 DataFrame
        
        # 添加線段的幾何資料
        line_gdf = gpd.GeoDataFrame(line_attributes, geometry=[line], crs=gdf.crs)
        
        # 將生成的線段添加到 line_output 列表
        line_output.append(line_gdf)

        # print(f"生成的線段 (ROADSEGID: {roadsegid}): {line}")

# 檢查生成的線段數量
# print(f"生成的線段數量: {len(line_output)}")

# 合併所有生成的線段到一個 GeoDataFrame
all_lines_gdf = gpd.GeoDataFrame(pd.concat(line_output, ignore_index=True))

# 將所有欄位的資料類型轉換為 shapefile 支援的類型
for col in all_lines_gdf.columns:
    if all_lines_gdf[col].dtype == 'int64':
        all_lines_gdf[col] = all_lines_gdf[col].astype(int)
    elif all_lines_gdf[col].dtype == 'float64':
        all_lines_gdf[col] = all_lines_gdf[col].astype(float)
    elif all_lines_gdf[col].dtype == 'object':
        all_lines_gdf[col] = all_lines_gdf[col].astype(str)

# 定義輸出文件路徑
output_shapefile = os.path.join(root, 'center_line.shp')

# 將生成的線段 GeoDataFrame 導出為 shapefile
all_lines_gdf.to_file(output_shapefile, driver='ESRI Shapefile', encoding='utf-8')

print(f"線段已成功匯出到 {output_shapefile}")

# 顯示合併後的 GeoDataFrame
# print("\n所有生成的線段:")
print(all_lines_gdf)







