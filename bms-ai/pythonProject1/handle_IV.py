import os
import pandas as pd
from tqdm import tqdm

def calculate_moving_average(series, window_size):
    """计算移动平均值"""
    return series.rolling(window=window_size, min_periods=1).mean()


def process_csv_files(folder_path):
    """处理文件夹中的所有CSV文件"""
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if file_path.endswith('.csv') and os.path.isfile(file_path):
            print(f'Processing {filename}')
            process_single_file(file_path)


def process_single_file(file_path):
    """处理单个CSV文件"""
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 假设CSV文件中有两列：时间和电流/电压
    current_column = df['Curr']
    voltage_column = df['Volt']

    # 计算每个时间步下前60个数据的平均电流和平均电压
    avg_current_60 = calculate_moving_average(current_column, 60)
    avg_voltage_60 = calculate_moving_average(voltage_column, 60)

    # 计算每个时间步下前300个数据的平均电流和平均电压
    avg_current_300 = calculate_moving_average(current_column, 300)
    avg_voltage_300 = calculate_moving_average(voltage_column, 300)

    # 计算每个时间步下前600个数据的平均电流和平均电压
    avg_current_600 = calculate_moving_average(current_column, 600)
    avg_voltage_600 = calculate_moving_average(voltage_column, 600)

    # 计算每个时间步下all数据的平均电流和平均电压
    L = len(current_column)
    avg_current_all = calculate_moving_average(current_column, L)
    avg_voltage_all = calculate_moving_average(voltage_column, L)




    # 将结果添加到DataFrame中
    df['avg_current60'] = avg_current_60
    df['avg_voltage60'] = avg_voltage_60
    df['avg_current300'] = avg_current_300
    df['avg_voltage300'] = avg_voltage_300
    df['avg_current600'] = avg_current_600
    df['avg_voltage600'] = avg_voltage_600
    df['avg_current_all'] = avg_current_all
    df['avg_voltage_all'] = avg_voltage_all

    # 保存结果回到原CSV文件
    df.to_csv(file_path, index=False)



# 使用示例
folder_path = 'D:/Code/code/submit/bms_ai/PycharmProjects/data/filter_data/data_handle_new'
process_csv_files(folder_path)