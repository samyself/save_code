import os
import pandas as pd


def check_and_delete_files(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)

            # 读取CSV文件
            data = pd.read_csv(file_path)

            # 假设电流和电压分别存储在 'current' 和 'voltage' 列中
            current = data['Curr'].values
            voltage = data['Volt'].values

            should_delete = False

            # 检查每个点的条件
            for i in range(1, len(current)):
                if ((current[i] - current[i - 1]) > -5 and (voltage[i] - voltage[i - 1]) < -15) or ((voltage[i] - voltage[i - 1])<-15 and (voltage[i+1] - voltage[i]>=0)) :
                    should_delete = True
                    break

            # 如果满足条件，删除文件
            if should_delete:
                os.remove(file_path)
                print(f"Deleted {filename} because it doesn't meet the criteria.")


# 替换为你要检查的文件夹路径
folder_path = 'D:/data/data_handle_new_test_over75_d0_skip_dot'
check_and_delete_files(folder_path)