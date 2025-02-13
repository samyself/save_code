import os

def delete_smaller_than_specified_sized_csv_files(directory, max_size=16 * 1):  # 默认最大大小为1KB (16 * 1024 bytes)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):  # 只处理CSV文件
                filepath = os.path.join(root, file)
                try:
                    if os.path.getsize(filepath) < max_size:  # 检查文件大小是否小于1KB
                        os.remove(filepath)
                        print(f"Deleted CSV file: {filepath} (Size: {os.path.getsize(filepath)} bytes)")
                except OSError as e:
                    print(f"Error checking or deleting file {filepath}: {e}")

# 替换为你想要处理的目录路径
target_directory = 'D:/data/data_handle_no_merge'
delete_smaller_than_specified_sized_csv_files(target_directory)