import os
import pandas as pd

def concatenate_csv_files(input_directory, output_file):
    # 获取目录下所有的CSV文件
    csv_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]
    
    # 初始化一个空的DataFrame
    concatenated_df = pd.DataFrame()
    
    # 逐个读取CSV文件并拼接
    for csv_file in csv_files:
        file_path = os.path.join(input_directory, csv_file)
        df = pd.read_csv(file_path)
        concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)
    
    # 输出拼接好的DataFrame到新的CSV文件
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated CSV saved to {output_file}")

# 设置输入目录和输出文件名
# input_directory = 'dataFT\laytex\merge'  # 替换为实际CSV文件所在的目录


#input_directory_train= './20250310_data/train_data/1q-/clear/test/'  # 替换为实际CSV文件所在的目录
#output_file_train = './20250310_data/train_data/1q-/clear/test.csv'  # 输出文件名
#input_directory_train= './20250406_data/MJQ/train_data/test/'  # 替换为实际CSV文件所在的目录
#output_file_train = './20250406_data/MJQ/train_data/test/test.csv'  # 输出文件名
input_directory_train= './motion_0407/2/ZS/train/'  # 替换为实际CSV文件所在的目录
output_file_train = './motion_0407/2/ZS/train/train.csv'  # 输出文件名


concatenate_csv_files(input_directory_train, output_file_train)




# input_directory_test = 'dataFT/exp1/trainset/C3test/'  # 替换为实际CSV文件所在的目录
# output_file_test = 'dataFT/exp1/C3test.csv'  # 输出文件名

# input_directory_train= 'dataFT/exp1/trainset/C3train/'  # 替换为实际CSV文件所在的目录
# output_file_train = 'dataFT/exp1/C3train.csv'  # 输出文件名

# 调用函数拼接CSV文件
# concatenate_csv_files(input_directory_test, output_file_test)


# import torch
# print(torch.cuda.is_available())  # 如果返回True，则CUDA可用
# print(torch.version.cuda)  # 打印PyTorch使用的CUDA版本