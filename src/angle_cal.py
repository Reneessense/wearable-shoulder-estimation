# angle_calculation.py

import numpy as np
import pandas as pd


def define_coordinate(normal_vec_columns, df_o):
    x = []
    y = []
    z = []
    for _, row in df_o.iterrows():
        points = row[normal_vec_columns].values.astype(float).reshape(-1, 3) #0.XP，1.T8，2.SN，3.C7     #0.XP，1.T8，2.GH1，3.GH2



        origin = points[2]
        midpoint1 = (points[0] + points[1]) / 2  # XP和T8的中点
        midpoint2 = (points[2] + points[3]) / 2  # SN和C7的中点

        # 计算y轴向量
        y_axis = midpoint2 - midpoint1  # 向量a-b，b为起点a为终点

        # 计算z轴向量
        plane_points1 = points[2] - midpoint1
        plane_points2 = points[3] - midpoint1
        z_axis = np.cross(plane_points1, plane_points2)  # z轴方向确定方法：叉乘右手螺旋法则

        # 计算x轴向量
        x_axis = np.cross(y_axis, z_axis)

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        x.append(x_axis)
        y.append(y_axis)
        z.append(z_axis)

    return x, y, z


def define_coordinate_scapula(normal_vec_columns, df_o):
    x = []
    y = []
    z = []
    for _, row in df_o.iterrows():
        points = row[normal_vec_columns].values.astype(float).reshape(-1, 3) #0.AA，1.AI，2.TS

        origin = points[0]

        # 计算z轴向量
        z_axis = points[0] - points[2]  # 向量a-b，b为起点a为终点

        # 计算x轴向量
        x_axis = np.cross(points[1] - points[0], points[2] - points[0])

        # 计算y轴向量
        y_axis = np.cross(z_axis, x_axis)

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        x.append(x_axis)
        y.append(y_axis)
        z.append(z_axis)
    return x, y, z


def calculate_angle(vector_list1, vector_list2):
    angles_degrees = []
    for vector1, vector2 in zip(vector_list1, vector_list2):
        # print(vector1)
        # print(vector2)
        dot_product = np.dot(vector1, vector2)  # 计算两个向量的点积
        vector1 = np.linalg.norm(vector1)
        vector2 = np.linalg.norm(vector2)
        cos_angle = dot_product / (vector1 * vector2)  # 使用点积公式计算两个向量之间夹角的余弦值
        angle_radians = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # 计算余弦值对应的弧度角
        angle_degrees = np.degrees(angle_radians)  # 将弧度转换为度

        angles_degrees.append(angle_degrees)  # 将计算出的角度添加到列表

    return angles_degrees


def calculate_angle_xyz(x1, y1, z1, x2, y2, z2):
    x_angles = calculate_angle(x1, x2)
    y_angles = calculate_angle(y1, y2)
    z_angles = calculate_angle(z1, z2)
    # print(x_angles)
    return x_angles, y_angles, z_angles

def calculate_angle_elbow(normal_vec_columns, df_o):
    angles_degrees = []
    for _, row in df_o.iterrows():
        points = row[normal_vec_columns].values.astype(float).reshape(-1, 3)  #0.WX，1.WN，2.LE，3，ME，4.GH1，5.GH2

        midpoint1 = (points[0] + points[1]) / 2  # WX and WN midpoint
        midpoint2 = (points[2] + points[3]) / 2  # LE and ME midpoint
        midpoint3 = (points[4] + points[5]) / 2  # GH1 and GH2 midpoint

        vector1 = midpoint1 - midpoint2
        vector2 = midpoint3 - midpoint2

        dot_product = np.dot(vector1, vector2)  # 计算两个向量的点积
        vector1 = np.linalg.norm(vector1)
        vector2 = np.linalg.norm(vector2)
        cos_angle = dot_product / (vector1 * vector2)  # 使用点积公式计算两个向量之间夹角的余弦值
        angle_radians = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # 计算余弦值对应的弧度角。np.clip 确保余弦值在有效范围内，避免计算反余弦时出现数值错误。
        angle_degrees = np.degrees(angle_radians)  # 将弧度转换为度
        angles_degrees.append(angle_degrees)  # 将计算出的角度添加到列表
    return angles_degrees

def calculate_all_angles(df_o):

    # 建立胸廓坐标系
    normal_vec_columns_1 = ['XP', 'XP.1', 'XP.2', 'T8', 'T8.1', 'T8.2', 'SN', 'SN.1', 'SN.2', 'C7', 'C7.1', 'C7.2']
    x1, y1, z1 = define_coordinate(normal_vec_columns_1, df_o)

    # 建立肩胛骨坐标系
    normal_vec_columns_2 = ['AA', 'AA.1', 'AA.2', 'AI', 'AI.1', 'AI.2', 'TS', 'TS.1', 'TS.2']
    x2, y2, z2 = define_coordinate_scapula(normal_vec_columns_2, df_o)

    # 建立肱骨坐标系
    normal_vec_columns_3 = ['LE', 'LE.1', 'LE.2', 'ME', 'ME.1', 'ME.2', 'GH1', 'GH1.1', 'GH1.2', 'GH2', 'GH2.1', 'GH2.2']
    x3, y3, z3 = define_coordinate(normal_vec_columns_3, df_o)

    # 计算角度1, 2, 3 【肱骨坐标系和胸廓坐标系】
    df_o['angle1'],  df_o['angle2'], df_o['angle3'] = calculate_angle_xyz(x1, y1, z1, x3, y3, z3)
    # print(df_o)

    # 计算角度4, 5, 6 【肩胛骨坐标系和胸廓坐标系】
    df_o['angle4'],  df_o['angle5'], df_o['angle6'] = calculate_angle_xyz(x1, y1, z1, x2, y2, z2)

    # 计算角度7, 8, 9 【肱骨坐标系和肩胛骨坐标系】
    df_o['angle7'],  df_o['angle8'], df_o['angle9'] = calculate_angle_xyz(x2, y2, z2, x3, y3, z3)

    # 计算角度10 【elbow角度】
    normal_vec_columns_4 = ['WX', 'WX.1', 'WX.2', 'WN', 'WN.1', 'WN.2', 'LE', 'LE.1', 'LE.2', 'ME', 'ME.1', 'ME.2', 'GH1', 'GH1.1', 'GH1.2', 'GH2', 'GH2.1', 'GH2.2']
    df_o['angle10'] = calculate_angle_elbow(normal_vec_columns_4, df_o)

    return df_o
