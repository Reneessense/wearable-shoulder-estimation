import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from angle_cal import calculate_all_angles  #angle_cal_pxy是缺少C7 的坐标系
#from angle_cal_double import calculate_all_angles  #angle_cal_pxy是缺少C7 的坐标系
from read_opticla import read_optical_data
#from read_snesor_data import read_sensor_data
from read_sensor_6ch import read_sensor_data
#from read_sensor_16ch import read_sensor_data
from get_intersection_data import get_intersection_data #6sensor用get_intersection_data_pxy


def process_single_data_group(optical_filepath, sensor_filepath, output_angle_dir,output_dft_dir):
    # 读取光捕数据
    df_o = read_optical_data(optical_filepath)
    # 计算光捕角度

    df_o = calculate_all_angles(df_o) # 肩关节角度
    df_angle = df_o[['Frame','Time','angle1','angle2','angle3','angle4','angle5','angle6','angle7','angle8','angle9','angle10','angle11','angle12','angle13','angle14','angle15','angle16','angle17','angle18']]
    #df_angle = df_o[['Frame','Time','angle1','angle2','angle3','angle4','angle5','angle6','angle7','angle8','angle9']]
    
    # 输出角度数据
    angle_output_path = os.path.join(output_angle_dir, os.path.basename(optical_filepath).replace('.csv', 'angle.csv'))
    df_angle.to_csv(angle_output_path, index=False)
    
    # 读取传感器数据
    df_s, df_s_resampled = read_sensor_data(sensor_filepath)
    
    # 数据对齐（交集）
    datafinal = get_intersection_data(df_angle, df_s_resampled)
    
    # 合并数据写出
    final_output_path = os.path.join(output_dft_dir, os.path.basename(sensor_filepath).replace('.txt', 'dft.csv'))
    datafinal.to_csv(final_output_path, index=False)
    
    return datafinal


def batch_process(optical_dir, sensor_dir,  output_angle_dir, output_dft_dir):
    # 确保输出目录存在
    os.makedirs( output_angle_dir, exist_ok=True)
    os.makedirs( output_dft_dir, exist_ok=True)


    # optical_files = [f for f in os.listdir(optical_dir) if f.endswith('.csv')]
    # sensor_files = [f for f in os.listdir(sensor_dir) if f.endswith('.txt')]

    if not os.path.exists(optical_dir):
        print("指定的目录不存在")
        # return
    optical_files = os.listdir(optical_dir)
    sensor_files = os.listdir(sensor_dir)


    for optical_file in optical_files:
        for sensor_file in sensor_files:
            optical_filepath = os.path.join(optical_dir, optical_file)
            sensor_filepath = os.path.join(sensor_dir, sensor_file)
            datafinal = process_single_data_group(optical_filepath, sensor_filepath, output_angle_dir,output_dft_dir)
            print(f'Processed {optical_file} and {sensor_file}')

    # for optical_file in optical_files:
    #     sensor_file = optical_file.replace('.csv', '.txt')
    #     if sensor_file in sensor_files:
    #         optical_filepath = os.path.join(optical_dir, optical_file)
    #         sensor_filepath = os.path.join(sensor_dir, sensor_file)
    #         datafinal = process_single_data_group(optical_filepath, sensor_filepath, output_angle_dir,output_dft_dir)
    #         print(f'Processed {optical_file} and {sensor_file}')

if __name__ == "__main__":
    # 设定文件夹路径
    #OPTICAL_DATA_DIR = './20270710/opt/comp'      # 光捕数据
    #SENSOR_DATA_DIR = './20270710/ssr/comp'  # 传感器数据
    #OUTPUT_ANGLE_DIR = './20270710/angle'    # 角度输出文件夹
    #OUTPUT_DFT_DIR = './20270710/train_data/16sensor+18angle'    # 训练数据输出

    #OPTICAL_DATA_DIR = './20250406_data/QNC/opt/2'      # 光捕数据
    #SENSOR_DATA_DIR = './20250406_data/QNC/ssr/2'  # 传感器数据
    #OUTPUT_ANGLE_DIR = './20250406_data/QNC/angle'    # 角度输出文件夹
    #OUTPUT_DFT_DIR = './20250406_data/QNC/train_data/6sensor+10angle'    # 训练数据输出

    #OPTICAL_DATA_DIR = './20250406_data/MJQ/opt'      # 光捕数据
    #SENSOR_DATA_DIR = './20250406_data/MJQ/ssr'  # 传感器数据
    #OUTPUT_ANGLE_DIR = './20250406_data/MJQ/angle'    # 角度输出文件夹
    #OUTPUT_DFT_DIR = './20250406_data/MJQ/train_data/6sensor+10angle'    # 训练数据输出

    OPTICAL_DATA_DIR = './20250310_data/optical/003'      # 光捕数据
    SENSOR_DATA_DIR = './20250310_data/sensor/003'  # 传感器数据
    OUTPUT_ANGLE_DIR = './20250310_data/angle'    # 角度输出文件夹
    OUTPUT_DFT_DIR = './20250310_data/train_data/6sensor+10angle'    # 训练数据输出

    #OPTICAL_DATA_DIR = './20250116/opt/User2B'      # 光捕数据
    #SENSOR_DATA_DIR = './20250116/sensor/User2B'  # 传感器数据
    #OUTPUT_ANGLE_DIR = './20250116/angle'    # 角度输出文件夹
    #OUTPUT_DFT_DIR = './20250116/train_data'    # 训练数据输出



batch_process (OPTICAL_DATA_DIR, SENSOR_DATA_DIR, OUTPUT_ANGLE_DIR, OUTPUT_DFT_DIR)