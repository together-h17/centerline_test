# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:16:20 2024

@author: ae133  角度判斷
"""

import geopandas as gpd
from shapely.geometry import Point, LineString
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

def azimuth(p1, p2):
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    azi = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    return azi

def angle(p1, p2, p3):
    azimuth1 = azimuth(p1, p2)
    azimuth2 = azimuth(p2, p3)
    ang = (180 - abs(azimuth2 - azimuth1))
    ang = min(ang, 360 - ang)
    return ang

# 定義路徑
root = r'D:\Users\ae133\Desktop\花蓮智慧國土\電子地圖中線\output'
# line_path = os.path.join(root, 'J_ROAD.shp')
line_path = r"D:\Users\ae133\Desktop\花蓮智慧國土\電子地圖中線\U_ROAD_Clip_single.shp"
# polygon_path = os.path.join(root, 'J_ROADA.shp')
polygon_path = r"D:\Users\ae133\Desktop\花蓮智慧國土\電子地圖中線\U_ROADA_Clip.shp"

output_nearest_points_path = os.path.join(root, 'nearest_points.shp')
output_second_nearest_points_path = os.path.join(root, 'second_nearest_points.shp')
output_combined_path = os.path.join(root, 'combined_output_v7.shp')
output_line_shapefile = os.path.join(root, 'output_line_v7.shp')


line_gdf = gpd.read_file(line_path, encoding='utf-8')
polygon_gdf = gpd.read_file(polygon_path)


nodes = []
nearest_points = []
second_nearest_points = []
crossover = []
midpoints = []
excluded_points = []  # 儲存所有被排除的點資料

# 添加新的屬性欄位 'ID'，並依序從 1 編號
line_gdf['ID'] = range(1, len(line_gdf) + 1)

# 將每條線的所有節點轉為點資料，保留屬性資料
for idx, line in tqdm(line_gdf.iterrows(), total=line_gdf.shape[0]):
   coords = list(line.geometry.coords)
   num_coords = len(coords)
   for p_idx, coord in enumerate(coords):
       point = Point(coord)
       line_attrs = line.drop('geometry').to_dict() # 為何要drop？
       # 設定 p_number，起點為1，終點為 num_coords
       line_attrs['p_number'] = p_idx + 1
       nodes.append({**line_attrs, 'geometry': point}) # **是什麼意思？

# 建立節點的 GeoDataFrame
nodes_gdf = gpd.GeoDataFrame(nodes, geometry='geometry', crs=line_gdf.crs)

# 創建面資料的 -0.01 公尺buffer
buffered_polygon = polygon_gdf.geometry.buffer(-0.01)
merged_buffer = buffered_polygon.unary_union # unary_union 與 union 的差別？

# 檢查點資料是否在新緩衝區之外
outside_points = nodes_gdf[~nodes_gdf.geometry.within(merged_buffer)] # ~是什麼意思？panadas

# 創建 5 公尺的緩衝區，排除範圍內的點
buffered_outside_points = outside_points.geometry.buffer(5) # 為什麼要這一行？
points_to_exclude = set() # set 資料型態特色？

# 選取排除緩衝區內的點
for idx, point in outside_points.iterrows():
   points_to_exclude.update(nodes_gdf[nodes_gdf.geometry.intersects(point.geometry.buffer(5))].index)

# 保存被排除的點資料
excluded_points = nodes_gdf.loc[list(points_to_exclude)].copy()

# 排除這些點
filtered_gdf = nodes_gdf.drop(list(points_to_exclude))

# 將座標作為字串，用於分組篩選
filtered_gdf['coord_str'] = filtered_gdf['geometry'].apply(lambda geom: f"{geom.x}_{geom.y}")

# 統計每個座標的出現次數
coordinate_counts = {}
for idx, row in filtered_gdf.iterrows():
   coord = (round(row.geometry.x, 5), round(row.geometry.y, 5))
   if coord in coordinate_counts:
       coordinate_counts[coord] += 1
   else:
       coordinate_counts[coord] = 1

# 找到出現次數超過 3 的座標並創建 buffer  # 排除路口點
points_to_exclude_counts = set()  
for coord, count in coordinate_counts.items():
   if count >= 3:
       # 創建 10m 的 buffer
       buffer_area = Point(coord).buffer(10)
       # 找出在緩衝區內的所有點
       points_to_exclude_counts.update(filtered_gdf[filtered_gdf.geometry.intersects(buffer_area)].index)

# 將被排除的點資料儲存
excluded_points = pd.concat([excluded_points, filtered_gdf.loc[list(points_to_exclude_counts)]], ignore_index=True)

# 再次排除這些點
filtered_gdf = filtered_gdf.drop(list(points_to_exclude_counts)) 

# 定義計算最近點的函數
def get_nearest_point_on_polygon(point, polygon):
    """找到點到 Polygon 或 MultiPolygon（包含外部和內部邊界）的最近點。"""
    
    # 判斷是否為 MultiPolygon，將其展開為多邊形的列表
    geometries = polygon.geoms if polygon.geom_type == 'MultiPolygon' else [polygon]
    
    nearest_point, min_distance = None, np.inf
    
    for poly in geometries:
        # exterior
        candidate_point = poly.exterior.interpolate(poly.exterior.project(point))
        distance = point.distance(candidate_point)
        if distance < min_distance:
            min_distance = distance
            nearest_point = candidate_point
            
        # interiors
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
  
     # 計算方位角

    azi = azimuth(closest_point, node.geometry)
    node_attrs = node.drop('geometry').to_dict()
    node_attrs['azimuth'] = azi
    nearest_points.append({**node_attrs, 'geometry': closest_point})
    
    # 計算180度旋轉後的方位角
    opposite_azimuth = (azi + 180) % 360
    
    # 嘗試找到第二最近點 # 與其說第二近，更像是找對側點(這段我不太清楚)
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
        nsp_distance = second_closest_point.distance(closest_point)
        if second_closest_point != closest_point and nsp_distance > step_size:
            break
        step_size += min_distance / 2
    
    # 抓交集中線次數    
    crossline = LineString([(closest_point.x, closest_point.y), (second_closest_point.x, second_closest_point.y)])
    crossover = line_gdf.intersection(crossline)
    crossover_count = sum(1 for geom in crossover if not geom.is_empty)
    
    second_attrs = node_attrs.copy()
    second_attrs['azimuth'] = opposite_azimuth
    second_attrs['crossover'] = crossover_count
    
    # 判斷同點
    second_attrs['samepoint'] = int(second_closest_point.intersects(closest_point.buffer(0.001)))
    geometry_add = node.geometry if (second_attrs['samepoint'] or crossover_count != 1) else second_closest_point
    second_nearest_points.append({**second_attrs, 'geometry': geometry_add})

    
# 建立中點資料
for npt, sp in zip(nearest_points, second_nearest_points):
    if sp['samepoint'] == 0 and sp['crossover'] == 1:
        midpoint_geom = Point(
            (npt['geometry'].x + sp['geometry'].x) / 2,
            (npt['geometry'].y + sp['geometry'].y) / 2
        )
    else:
        midpoint_geom = sp['geometry']
        
    midpoint_attrs = {**npt, **sp}  # 保留所有屬性資料
    midpoints.append({**midpoint_attrs, 'geometry': midpoint_geom})


# 將排除的點資料與中點資料合併
combined_points = pd.concat([pd.DataFrame(midpoints), excluded_points], ignore_index=True)
combined_points = combined_points.drop(columns=['coord_str', 'azimuth'], errors='ignore')
# 將合併後的點資料轉換為 GeoDataFrame
combined_points_gdf = gpd.GeoDataFrame(combined_points, geometry='geometry', crs=filtered_gdf.crs)

# 儲存結果
combined_points_gdf.to_file(output_combined_path, encoding='utf-8')  # 儲存中點和排除的點資料
# gpd.GeoDataFrame(nearest_points).to_file(output_nearest_points_path, encoding='utf-8')  # 儲存最近點
# gpd.GeoDataFrame(second_nearest_points).to_file(output_second_nearest_points_path, encoding='utf-8')  # 儲存第二最近點

# 根據 ID 分組生成線段
grouped = combined_points_gdf.groupby('ID')
line_output = []
for roadsegid, group in grouped:
    group = group.sort_values(by='p_number')
    if len(group) > 1:
        angle_geometries = []
        if len(group) > 2:
             # 角度判斷
            for i in range(1, len(group) - 1):  # 頭尾不算
                p1 = group.iloc[i - 1].geometry
                p2 = group.iloc[i].geometry
                p3 = group.iloc[i + 1].geometry
                ang = angle(p1, p2, p3)
                if ang < 80:
                    old_node = nodes_gdf[(nodes_gdf['ID'] == roadsegid) & (nodes_gdf['p_number'] == i+1)]
                    angle_geometries.append(old_node.geometry.iloc[0])
                else:
                    angle_geometries.append(p2)
        angle_geometries.insert(0, group.iloc[0].geometry) # insert
        angle_geometries.append(group.iloc[-1].geometry)
        
        line = LineString(angle_geometries)
        first_point_attributes = group.iloc[0].drop(['geometry', 'p_number', 'ID']).fillna("")
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