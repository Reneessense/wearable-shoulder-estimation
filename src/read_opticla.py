import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import re

def convert_to_24hr_format(time_str: str) -> str:
    """
    将时间字符串转换为24小时制格式。
    """
    if '下午' in time_str:
        time_obj = datetime.strptime(time_str, '%Y-%m-%d %I.%M.%S.%f 下午')
        if time_obj.hour < 12:
            time_obj = time_obj.replace(hour=time_obj.hour + 12)
    else:
        time_obj = datetime.strptime(time_str, '%Y-%m-%d %I.%M.%S.%f 上午')
    
    return time_obj.strftime('%Y-%m-%d %H:%M:%S.%f')

def read_optical_data(filepath: str, encoding='utf-8') -> pd.DataFrame:
    """
    读取光捕数据文件并进行处理。
    """
    with open(filepath, 'r', encoding=encoding) as f:
        first_row = f.readline().strip()
        second_row = f.readline().strip()
        third_row = f.readline().strip()
        fourth_row = f.readline().strip()

    print(f"First row: {first_row}")
    print(f"Second row: {second_row}")
    print(f"Third row: {third_row}")
    print(f"Fourth row: {fourth_row}")
    
    # 解析开始时间
    columns = first_row.split(',')
    capture_start_time = columns[11]
    capture_start_time_24hr = convert_to_24hr_format(capture_start_time)
    start_time = pd.to_datetime(capture_start_time_24hr, format='%Y-%m-%d %H:%M:%S.%f')

    # 跳过前7行读取数据
    df_o = pd.read_csv(filepath, header=None, delimiter=",", low_memory=False, encoding=encoding, skiprows=7)

    # 构造新列名
    new_columns = ['Frame', 'Time']
    fourth_row_columns = fourth_row.split(',')
    for col in fourth_row_columns[2:]:
        #if "Shoulder:" in col:
        if "MarkerSet 0710_shoulder_double_side:" in col:
            new_columns.append(col.split(":")[-1])
        else:
            new_columns.append(col)
    df_o.columns = new_columns

    # 构造新列名，统一将含 ":" 的列名简化为冒号后的短名
    #new_columns = ['Frame', 'Time']
    #fourth_row_columns = fourth_row.split(',')

    #for col in fourth_row_columns[2:]:
    #    if ":" in col:
    #        new_columns.append(col.split(":")[-1])
    #    else:
    #        new_columns.append(col)

    # 去重处理
    name_count = {}
    new_column_names = []
    for col in df_o.columns:
        if col in name_count:
            name_count[col] += 1
            new_column_names.append(f"{col}.{name_count[col]}")
        else:
            name_count[col] = 0
            new_column_names.append(col)
    df_o.columns = new_column_names

    # 添加绝对时间列
    df_o["AbsTime"] = df_o["Time"].astype(float).apply(lambda x: start_time + timedelta(seconds=x))

    # 转换数据列为 float
    col_num = len(df_o.columns)
    df_o1 = df_o.iloc[:, 2:col_num - 1].astype(float)
    df_o2 = df_o[["Frame", "AbsTime"]]
    df_o = pd.concat([df_o2, df_o1], axis=1)
    df_o.rename(columns={"AbsTime": "Time"}, inplace=True)

    return df_o

# ✅ 测试文件路径（你需根据实际路径替换）
#file_path = './20250406_data/MJQ/opt/mjq0403.csv'
file_path = './20270710/opt/comp/opt.csv'

df = read_optical_data(file_path)
print(df.head())
