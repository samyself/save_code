import os
import pandas as pd


def delete_files_with_current_spike(folder_path):
    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for file in files:
        file_path = os.path.join(folder_path, file)

        # 读取CSV文件
        data = pd.read_csv(file_path)

        # 假设电流数据存储在某一列中，例如'current'
        current_data = data['Curr'].values

        # 初始化标志位，用于标记是否需要删除文件
        has_current_spike = False

        # 遍历电流数据，查找是否有过电流上升大于10A的情况
        for i in range(1, len(current_data)):
            if current_data[i] - current_data[i - 1] >6:
                has_current_spike = True
                break

        # 如果出现过电流上升大于10A的情况，则删除该文件
        if has_current_spike:
            os.remove(file_path)
            print(f"Deleted file {file} because it contains a spike in current.")


# 使用函数
folder_path = 'D:/data/data_handle_SOC_over70_no_diff_skip_dot'
delete_files_with_current_spike(folder_path)