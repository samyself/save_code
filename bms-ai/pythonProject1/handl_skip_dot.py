import os
import pandas as pd


def extract_every_fifth_row(input_file, output_file):
    # 读取输入的CSV文件
    data = pd.read_csv(input_file)

    # 检查是否有足够的行数
    if len(data) < 5:
        print(f"Warning: {input_file} does not have enough rows to extract every fifth row.")
        return

    # 创建一个新的DataFrame来存储结果
    result = pd.DataFrame()

    # 每隔5个数提取一行数据
    result['time'] = data['time'][::5]
    result['Curr'] = data['Curr'][::5]
    result['Volt'] = data['Volt'][::5]
    result['Temp'] = data['Temp'][::5]
    result['TrueSoc'] = data['TrueSoc'][::5]
    result['BmsSoc'] = data['BmsSoc'][::5]
    result['Ah'] = data['Ah'][::5]
    result['avg_current300'] = data['avg_current300'][::5]
    result['avg_voltage300'] = data['avg_voltage300'][::5]
    result['avg_current'] = data['avg_current'][::5]
    result['avg_voltage'] = data['avg_voltage'][::5]
    # 保存结果到新的CSV文件
    result.to_csv(output_file, index=False)


def process_folders(input_folder, output_folder):
    # 遍历输入文件夹中的所有子文件夹和文件
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv'):
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_file_path, input_folder)
                output_file_path = os.path.join(output_folder, relative_path)

                # 处理CSV文件
                extract_every_fifth_row(input_file_path, output_file_path)


# 指定输入文件夹路径和输出文件夹路径
input_folder = 'D:/data/data_handle_SOC_over70_d0'
output_folder = 'D:/data/data_handle_SOC_over70_d0_skip_dot'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# 调用函数
process_folders(input_folder, output_folder)