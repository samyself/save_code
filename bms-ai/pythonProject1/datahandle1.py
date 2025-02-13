# 导入需要用的包
import pandas as pd
import numpy as np
from pathlib import Path
import os

def data_prepare1(T):
    # 取SOC的起点
    T_each_all = pd.DataFrame(
        {'time': [], 'TrueSoc': [], 'Volt': [], 'Temp': [], 'Curr': [], 'BmsSoc': [], 'Ah': []})
    idx_begin = 0
    for i in range(len(T['TrueSoc'])):
        if T['TrueSoc'][i] >= 70:
            idx_begin = i
            break
    T = T.iloc[idx_begin:-1, :].reset_index(drop=True)
    T_each_all = pd.DataFrame({
        'time': T['time'],
        'TrueSoc': T['TrueSoc'],
        'Volt': T['Volt'],
        'Temp': T['Temp'],
        'Curr': T['Curr'],
        'BmsSoc': T['BmsSoc'],
        'Ah': T['Ah'],
        #'avg_current':T['avg_current'],
        #'avg_voltage':T['avg_voltage']
    })
    return T_each_all

# 主函数
def main():
    input_folder_path = 'D:/data/data_handle_new'  # 替换为你的输入文件夹路径
    output_folder_path = 'D:/data/data_handle_SOC_over70'  # 替换为你想要保存处理后文件的文件夹路径

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    i = 0
    for filename in os.listdir(input_folder_path):
        if filename.endswith('.csv') & (i >-1):
            i = i + 1
            file_path = os.path.join(input_folder_path, filename)

            # 读取CSV文件
            df = pd.read_csv(file_path)

            #处理数据
            processed_df  = data_prepare1(df)
            # 构造输出文件名
            output_filename = f"processed_SOC_70_{filename}"
            output_file_path = os.path.join(output_folder_path, output_filename)

            # 保存处理后的数据到新文件
            if (len(processed_df) > 0):
                processed_df.to_csv(output_file_path, index=False)
                print(f"Processed and saved {filename} to {output_filename}")
            else:
                print(filename)

if __name__ == "__main__":
    main()