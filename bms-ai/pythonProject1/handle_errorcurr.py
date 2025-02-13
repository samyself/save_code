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

        # 初始化标志位，用于标记是否需要删除文件
        should_delete = False

        # 遍历电流数据，查找满足条件的点
        for i in range(len(current_data)):
            if i > 0:
                # 检查当前点和前一点的电流差值
                if current_data[i - 1] - current_data[i] > 10:
                    # 查找后续60个数据点内电流是否上升超过10A,即1分钟以内出现电流跳变
                    for j in range(i + 1, min(i + 12, len(current_data))):
                        if current_data[j] - current_data[i] > 10:
                            should_delete = True
                            break
            if should_delete:
                break

        # 如果需要删除文件，则执行删除操作
        if should_delete:
            os.remove(file_path)
            print(f"Deleted file {file} because it contains sudden changes in current.")

# 使用函数
folder_path = 'D:/data/data_handle_SOC_over70_no_diff_skip_dot'
delete_files_with_sudden_changes(folder_path)