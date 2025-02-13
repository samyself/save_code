import os
import pandas as pd


def delete_files_with_sudden_changes(folder_path):
    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for file in files:
        file_path = os.path.join(folder_path, file)

        # 读取CSV文件
        data = pd.read_csv(file_path)

        # 假设电流数据存储在某一列中，例如'current'
        current_data = data['Curr'].values
        if current_data[0]<=100:
            os.remove(file_path)
            print(f"Deleted file {file} because it contains sudden changes in current.")

# 使用函数
folder_path = 'D:/data/data_handle_new_test_over75_d0_skip_dot_big_curr'
delete_files_with_sudden_changes(folder_path)