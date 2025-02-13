# 导入需要用的包
import pandas as pd
import numpy as np
from pathlib import Path
import os
def contains_nonpositive(file_path):
    """检查CSV文件中是否存在小于等于0的值"""
    df = pd.read_csv(file_path)
    nonpositive_values = df <= 0
    return nonpositive_values.any().any()

def remove_bad_files(folder_path):
    """遍历文件夹并删除含有小于等于0值的CSV文件"""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                if contains_nonpositive(file_path):
                    print(f"检测到文件 {file_path} 中含有小于等于0的值，正在删除...")
                    os.remove(file_path)
                    print(f"已删除文件: {file_path}")
                else:
                    print(f"文件 {file_path} 不含小于等于0的值，跳过处理")

folder_path = "D:/data/data_handle_new_test_over70_d0_skip_dot"
remove_bad_files(folder_path)