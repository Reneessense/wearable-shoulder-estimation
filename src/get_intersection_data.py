import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def get_intersection_data(df_angle: pd.DataFrame, df_s: pd.DataFrame) -> pd.DataFrame:
    """
    获取光捕和传感器数据的交集，并按时间对齐合并。
    
    参数:
        df_angle: 光捕数据，要求有 'Time' 列，且为 datetime 类型。
        df_s: 传感器数据，要求 index 是 datetime 类型。
        
    返回:
        合并后的交集 DataFrame，包括时间戳、传感器值和角度值。
    """

    # 类型检查输出
    print("🧪 df_angle['Time'] 类型：", type(df_angle['Time'].iloc[0]) if 'Time' in df_angle.columns else '不存在')
    print("🧪 df_s index 类型：", type(df_s.index[0]))

    # ✅ 计算交集起止时间（都必须是 Timestamp 类型）
    s1 = max(df_angle['Time'].min(), df_s.index.min())
    e1 = min(df_angle['Time'].max(), df_s.index.max())

    # ✅ 设置光捕数据的时间为 index（传感器数据已在读取时设为 index）
    df_angle = df_angle.copy().set_index('Time')
    df_angle = df_angle[s1:e1]
    df_angle.insert(0, 'Time_angle', df_angle.index)

    df_s = df_s.copy()
    df_s = df_s[s1:e1]
    df_s.insert(0, 'Time_sensor', df_s.index)

    # ✅ 打印交集信息
    print(f'交集起始：\t\t{s1}')
    print(f'交集结束：\t\t{e1}')
    print(f'交集总时长(s)：\t{(e1 - s1).total_seconds()}')
    print(f'交集帧数：\t\t光捕 {len(df_angle)}，传感器 {len(df_s)}')

    # ✅ 合并两组数据（时间点最近匹配）
    merged_df = pd.merge_asof(df_s, df_angle, left_index=True, right_index=True, direction='nearest')

    # ✅ 合并检查
    nan_count = merged_df['Frame'].isna().sum() if 'Frame' in merged_df.columns else 0
    print(f'匹配成功的数量: {len(merged_df) - nan_count}')

    # ✅ 添加时间差列（这里先设为 0，可改为实际 delta）
    merged_df.insert(0, 'Time_delta', 0.0)

    # ✅ 将 Time_angle 列移至第3列
    merged_df.insert(2, 'Time_angle', merged_df.pop('Time_angle'), allow_duplicates=False)

    # ✅ 最终字段选择（你可以按需修改字段顺序）
    #datafinal = merged_df[['Time_angle', 's1', 's2', 's3', 's4', 's5', 's6',
    #                       'angle1', 'angle2', 'angle3', 'angle4', 'angle5',
    #                       'angle6', 'angle7', 'angle8', 'angle9', 'angle10']]
    
    datafinal = merged_df[['Time_angle',
                       's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
                       's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16',
                       'angle1', 'angle2', 'angle3', 'angle4', 'angle5', 'angle6',
                       'angle7', 'angle8', 'angle9', 'angle10', 'angle11', 'angle12',
                       'angle13', 'angle14', 'angle15', 'angle16', 'angle17', 'angle18']]


    datafinal.reset_index(drop=True, inplace=True)
    return datafinal
