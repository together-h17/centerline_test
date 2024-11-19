# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:43:49 2024

@author: AC512
"""
import geopandas as gpd 
from shapely.geometry import Point, LineString
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

# 定義路徑
root = r'D:\道路中線測試\花蓮output'
line_path = r"D:\道路中線測試\花蓮中線測試資料\U_ROAD_Clip_single.shp"
polygon_path = r"D:\道路中線測試\花蓮中線測試資料\U_ROADA_Clip.shp"
output_nearest_points_path = os.path.join(root, 'nearest_points.shp')
output_second_nearest_points_path = os.path.join(root, 'second_nearest_points.shp')
output_combined_path = os.path.join(root, 'combined_output.shp')
output_line_shapefile = os.path.join(root, 'output_line.shp')

# 讀取資料
line_gdf = gpd.read_file(line_path, encoding='utf-8')
polygon_gdf = gpd.read_file(polygon_path)

# 初始化資料儲存列表
nodes = []
nearest_points = []
second_nearest_points = []
midpoints = []
excluded_points = []  # 儲存所有被排除的點資料

# 將每條線的所有節點轉為點資料，保留屬性資料
for idx, line in tqdm(line_gdf.iterrows(), total=line_gdf.shape[0]):
    coords = list(line.geometry.coords)
    num_coords = len(coords)
    for p_idx, coord in enumerate(coords):
        point = Point(coord)
        line_attrs = line.drop('geometry').to_dict()
        line_attrs['p_number'] = p_idx + 1
        nodes.append({**line_attrs, 'geometry': point}) 

# 建立節點的 GeoDataFrame
nodes_gdf = gpd.GeoDataFrame(nodes, geometry='geometry', crs=line_gdf.crs)

# 創建面資料的 -0.01 公尺buffer
buffered_polygon = polygon_gdf.geometry.buffer(-0.01)
merged_buffer = buffered_polygon.unary_union

# 檢查點資料是否在新緩衝區之外
outside_points = nodes_gdf[~nodes_gdf.geometry.within(merged_buffer)]

# 創建 5 公尺的緩衝區，排除範圍內的點
buffered_outside_points = outside_points.geometry.buffer(5)
points_to_exclude = set()
for idx, point in outside_points.iterrows():
    points_to_exclude.update(nodes_gdf[nodes_gdf.geometry.intersects(point.geometry.buffer(5))].index)

# 保存被排除的點資料
excluded_points = nodes_gdf.loc[list(points_to_exclude)].copy()

# 排除這些點
filtered_gdf = nodes_gdf.drop(list(points_to_exclude))

# 統計每個座標的出現次數，並排除出現次數超過 3 的座標
filtered_gdf['coord_str'] = filtered_gdf['geometry'].apply(lambda geom: f"{geom.x}_{geom.y}")
coordinate_counts = {}
for idx, row in filtered_gdf.iterrows():
    coord = (round(row.geometry.x, 5), round(row.geometry.y, 5))
    if coord in coordinate_counts:
        coordinate_counts[coord] += 1
    else:
        coordinate_counts[coord] = 1
points_to_exclude_counts = set()  
for coord, count in coordinate_counts.items():
    if count >= 3:
        buffer_area = Point(coord).buffer(10)
        points_to_exclude_counts.update(filtered_gdf[filtered_gdf.geometry.intersects(buffer_area)].index)

# 將被排除的點資料儲存
excluded_points = pd.concat([excluded_points, filtered_gdf.loc[list(points_to_exclude_counts)]], ignore_index=True)

# 再次排除這些點
filtered_gdf = filtered_gdf.drop(list(points_to_exclude_counts))

# 定義計算最近點的函數
def get_nearest_point_on_polygon(point, polygon):
    geometries = polygon.geoms if polygon.geom_type == 'MultiPolygon' else [polygon]
    nearest_point, min_distance = None, np.inf
    for poly in geometries:
        candidate_point = poly.exterior.interpolate(poly.exterior.project(point))
        distance = point.distance(candidate_point)
        if distance < min_distance:
            min_distance = distance
            nearest_point = candidate_point
        for interior in poly.interiors:
            candidate_point = interior.interpolate(interior.project(point))
            distance = point.distance(candidate_point)
            if distance < min_distance:
                min_distance = distance
                nearest_point = candidate_point
    return nearest_point

# 找到最近點
for idx, node in tqdm(filtered_gdf.iterrows(), total=filtered_gdf.shape[0]):
    min_distance = np.inf
    closest_point = node.geometry
    for _, polygon in polygon_gdf.iterrows():
        nearest = get_nearest_point_on_polygon(node.geometry, polygon.geometry)
        distance = node.geometry.distance(nearest)
        if distance < min_distance:
            min_distance = distance
            closest_point = nearest

    dx = closest_point.x - node.geometry.x
    dy = closest_point.y - node.geometry.y
    azimuth = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    node_attrs = node.drop('geometry').to_dict()
    node_attrs['azimuth'] = azimuth
    nearest_points.append({**node_attrs, 'geometry': closest_point})
   
    opposite_azimuth = (azimuth + 180) % 360
    step_size = min_distance
    second_closest_point = None
    for _ in range(100):
        new_x = node.geometry.x + step_size * np.sin(np.radians(opposite_azimuth))
        new_y = node.geometry.y + step_size * np.cos(np.radians(opposite_azimuth))
        second_point = Point(new_x, new_y)
        min_distance = np.inf
        for _, polygon in polygon_gdf.iterrows():
            nearest = get_nearest_point_on_polygon(second_point, polygon.geometry)
            distance = second_point.distance(nearest)
            if distance < min_distance:
                min_distance = distance
                second_closest_point = nearest
        if second_closest_point != closest_point:
            break
        step_size += min_distance / 2
    second_attrs = node_attrs.copy()
    second_attrs['azimuth'] = opposite_azimuth
    second_nearest_points.append({**second_attrs, 'geometry': second_closest_point})

# 建立中點資料
for npt, sp in zip(nearest_points, second_nearest_points):
    midpoint_geom = Point(
        (npt['geometry'].x + sp['geometry'].x) / 2,
        (npt['geometry'].y + sp['geometry'].y) / 2
    )
    midpoint_attrs = {**npt, **sp}
    midpoints.append({**midpoint_attrs, 'geometry': midpoint_geom})

# 將排除的點資料與中點資料合併
combined_points = pd.concat([pd.DataFrame(midpoints), excluded_points], ignore_index=True)
combined_points = combined_points.drop(columns=['coord_str', 'azimuth'], errors='ignore')
combined_points_gdf = gpd.GeoDataFrame(combined_points, geometry='geometry', crs=filtered_gdf.crs)
combined_points_gdf.to_file(output_combined_path, encoding='utf-8')

# 根據 ROADSEGID 分組，生成線段
grouped = combined_points_gdf.groupby('ROADSEGID')
line_output = []
for roadsegid, group in grouped:
    group = group.sort_values(by='p_number')
    if len(group) > 1:
        line = LineString(group.geometry.tolist())
        first_point_attributes = group.iloc[0].drop(['geometry', 'p_number']).fillna("")
        line_attributes = first_point_attributes.to_frame().T
        line_gdf = gpd.GeoDataFrame(line_attributes, geometry=[line], crs=combined_points_gdf.crs)
        line_output.append(line_gdf)

# 合併並儲存所有生成的線段
all_lines_gdf = gpd.GeoDataFrame(pd.concat(line_output, ignore_index=True))
for col in all_lines_gdf.columns:
    if all_lines_gdf[col].dtype == 'int64':
        all_lines_gdf[col] = all_lines_gdf[col].astype(int)
    elif all_lines_gdf[col].dtype == 'float64':
        all_lines_gdf[col] = all_lines_gdf[col].astype(float)
    elif all_lines_gdf[col].dtype == 'object':
        all_lines_gdf[col] = all_lines_gdf[col].astype(str)
all_lines_gdf.to_file(output_line_shapefile, driver='ESRI Shapefile', encoding='utf-8')

print("中點、被排除的點資料、最近點及第二最近點已成功生成並輸出！")
print("線段已成功匯出到", output_line_shapefile)

