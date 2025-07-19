import serial
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from model import LSTM

# ✅ 1. 设置串口连接参数
SERIAL_PORT = "COM14"
BAUD_RATE = 115200
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
print(f"✅ 已连接到 {SERIAL_PORT}")

# ✅ 2. 设置 LSTM 模型参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 6
hidden_size = 128
num_layers = 2
output_size = 9
dropout = 0.3

# ✅ 3. 加载 LSTM 预测模型
model = LSTM(input_size, hidden_size, num_layers, output_size, dropout).to(device)
model.load_state_dict(torch.load("./result/model.ckpt", map_location=device))
model.eval()

# ✅ 4. 定义数据存储
window_length = 125
step_size = 125
buffer = deque(maxlen=window_length)

# 加载 scaler 和 angle_scaler
with open('sensor_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('angle_scaler.pkl', 'rb') as f:
    scaler_angle = pickle.load(f)

# ✅ 5. Matplotlib实时绘制预测角度
plt.ion()
fig, ax = plt.subplots(figsize=(12, 8))
predicted_angle_history = [[] for _ in range(output_size)]
predicted_x = []
angle_lines = [ax.plot([], [], label=f'Angle {i+1}')[0] for i in range(output_size)]

ax.set_xlim(0, 250)
ax.set_ylim(0, 180)  # 根据实际角度调整
ax.set_title("Real-time Predicted Angles")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angle")
ax.legend()

print("🚀 开始实时预测")
frame_counter = 0
sensor_check_counter = 0

# ✅ 6. 实时读取数据、预测、绘图
while True:
    try:
        line = ser.readline().decode("latin1", errors="ignore").strip()
        if not line:
            continue

        values = np.array(line.split(","), dtype=float)

        if len(values) != 6:
            print(f"⚠️ 数据格式错误: {values}")
            continue

        buffer.append(values)
        frame_counter += 1

        # 打印前100条数据检查传感器
        if sensor_check_counter < 100:
            print(f"🛠️ 传感器数据 ({sensor_check_counter+1}/100): {values}")
            sensor_check_counter += 1

        # 每125帧（1秒）执行一次预测
        if len(buffer) == window_length and frame_counter % step_size == 0:
            input_data = scaler.transform(np.array(buffer))
            input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
            input_tensor = input_tensor.unsqueeze(0)

            with torch.no_grad():
                predicted_angles_norm = model(input_tensor).cpu().numpy()

            # 角度反归一化
            predicted_angles = scaler_angle.inverse_transform(predicted_angles_norm)[0]

            predicted_x.append(frame_counter // step_size)
            for i in range(output_size):
                predicted_angle_history[i].append(predicted_angles[i])

            # 控制绘图数据长度
            if len(predicted_x) > 250:
                predicted_x = predicted_x[-250:]
                for i in range(output_size):
                    predicted_angle_history[i] = predicted_angle_history[i][-250:]

            # 更新绘图
            for i, line in enumerate(angle_lines):
                line.set_xdata(predicted_x)
                line.set_ydata(predicted_angle_history[i])
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.001)

            print(f"🎯 第{frame_counter // step_size}秒预测真实角度: {predicted_angles}")

        sensor_check_counter += 1

    except KeyboardInterrupt:
        print("❌ 程序终止")
        break
    except Exception as e:
        print(f"⚠️ 发生错误: {e}")

ser.close()