import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# ✅ 设置输入路径
input_dir = './20250310_data/train_data/all_but_rest/10ts-angle7'
#input_dir = './motion_0407/MJQ/2es-angle9'
#input_dir = './motion_0407/QNC/2ws-angle8'


# ✅ 遍历该文件夹下的所有 CSV 文件
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_dir, filename)
        print(f"📂 正在处理文件: {filename}")

        # ✅ 读取 CSV
        df = pd.read_csv(file_path)

        # ✅ 检查是否有 angle2 列，动作6用
        if 'angle7' not in df.columns:
            print(f"⚠️ 文件 {filename} 中未找到 'angle7' 列，跳过")
            continue

        # ✅ 使用 angle2 取反，寻找谷值（极小点）
        angle_signal = df['angle4'].values
        valleys, _ = find_peaks(-angle_signal, distance=500)  # 控制周期最短距离

        if len(valleys) < 2:
            print(f"⚠️ 文件 {filename} 周期点不足，跳过")
            continue

        # ✅ 每两个谷值切出一个周期
        for i in range(len(valleys) - 1):
            start = valleys[i]
            end = valleys[i + 1]
            cycle_df = df.iloc[start:end].reset_index(drop=True)
       
       
       ## ✅ 提取角度信号并寻找满足高度条件的波峰,动作5使用
       #angle_signal = df['angle1'].values
       ##peaks, properties = find_peaks(angle_signal, height=(10, 25), distance=100) #5td
       #peaks, properties = find_peaks(angle_signal, height=(5, 15), distance=100)  #5ta

        #if len(peaks) < 2:
        #    print(f"⚠️ 文件 {filename} 波峰点不足，跳过")
        #    continue

        ## ✅ 每两个波峰切出一个周期
        #for i in range(len(peaks) - 1):
        #    start = peaks[i]
        #    end = peaks[i + 1]
        #    cycle_df = df.iloc[start:end].reset_index(drop=True)

            # ✅ 构建新文件名（原名_01.csv 等）
            base_name = os.path.splitext(filename)[0]
            new_name = f"{base_name}_{i+1:02d}.csv"
            new_path = os.path.join(input_dir, new_name)

            # ✅ 保存新周期段
            cycle_df.to_csv(new_path, index=False)
            print(f"✅ 周期段 {i+1} 已保存: {new_name}")

print("🎉 所有文件已完成周期切分！")
