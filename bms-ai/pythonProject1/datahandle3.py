import os
import pandas as pd
import numpy as npFalse



def has_sudden_change(data):

    change_rates = data.diff().abs()
    sudden_change_indices = change_rates[change_rates > 10].index
    for i in sudden_change_indices:
        if i+10<len(data)-1:
            if  abs(data[i-1]-data[i+10]) <=5:
                return True
    else:
        return
def delete_files_with_sudden_current_change(folder_path):
    """
    删除包含突然变化的电流数据的CSV文件。

    参数:
        folder_path (str): 包含CSV文件的文件夹路径。
        output_folder (str): 可选参数，用于存储未删除的文件。如果没有提供，则移动到回收站。
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            if has_sudden_change(df['Curr']):
                print(f"Deleting file {filename} due to sudden current change.")
                os.remove(file_path)
            else:
                print(f"Keeping file {filename} as it does not have a sudden current change.")


# 使用示例
folder_path = 'D:/data/data_handle_new_test_over70_d0_skip_dot'
delete_files_with_sudden_current_change(folder_path)