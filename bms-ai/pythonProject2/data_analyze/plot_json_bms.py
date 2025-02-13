import json
import os
import shutil


def parse_json_arrays_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        buffer = ''
        A = []
        for line in file:
            buffer += line
            # 检查是否到达一个完整的 JSON 数组
            if buffer.endswith(']\n'):
                try:
                    # 尝试解析当前缓冲区的内容
                    parsed_array = json.loads(buffer)
                    A.extend(parsed_array)
                    # yield parsed_array
                    buffer = ''  # 清空缓冲区，准备下一个数组
                except json.JSONDecodeError as e:
                    print(f"Failed to decode JSON: {e}")
                    # 如果解析失败，尝试从缓冲区中移除最后一个字符并继续
                    buffer = buffer[:-1]
    print(len(A))
    return A


def cp_file(file_path, save_fold):
    try:
        shutil.copy2(file_path, save_fold)
    except Exception as e:
        print(f"{file_path} \nError: {e}")


# 使用示例
file_path = '../../data/json_res/analyze_info_Soc_0102_lstm_v1_2_delta.json'  # 替换为你的文件路径
data = parse_json_arrays_from_file(file_path)

# save_fold_890 = 'D:/Code/code/submit/bms_ai/PycharmProjects/data/pic/1204_lstmv5k_error_pic890'
# save_fold_910 = 'D:/Code/code/submit/bms_ai/PycharmProjects/data/pic/1204_lstmv5k_error_pic910'
# save_fold_910_high = 'D:/Code/code/submit/bms_ai/PycharmProjects/data/pic/1204_lstmv5k_error_pic910_high'
# save_fold_890_high = 'D:/Code/code/submit/bms_ai/PycharmProjects/data/pic/1204_lstmv5k_error_pic890_high'
# if not os.path.exists(save_fold_890):
#     os.makedirs(save_fold_890)
# if not os.path.exists(save_fold_910):
#     os.makedirs(save_fold_910)
# if not os.path.exists(save_fold_890_high):
#     os.makedirs(save_fold_890_high)
# if not os.path.exists(save_fold_910_high):
#     os.makedirs(save_fold_910_high)
num90_100 = 0
num90_100_high = 0
num80_90 = 0
num80_90_high = 0
num_80_90_max_good = 0
num_90_100_max_good = 0
num_80_90_mae_good = 0
num_90_100_mae_good = 0

for json in data:
    if '90.0_100.0' in json['file_info']:
        if json['soc_br_max'] >= 3.0:
            # download_name = json['file_info'].replace('_90.0_100.0','')
            # base_fold = 'D:/Code/code/submit/bms_ai/PycharmProjects/data/pic/Soc_1204_lstm_v5k_90.0_100.0'
            # input_path = f'{base_fold}/{download_name}.png'
            # save_path = f'{save_fold_910}/{download_name}.png'
            # cp_file(input_path, save_path)
            num90_100 += 1
            print(json['file_info'])
        if json['soc_br_max'] >= 2.0:
            # download_name = json['file_info'].replace('_90.0_100.0','')
            # base_fold = 'D:/Code/code/submit/bms_ai/PycharmProjects/data/pic/Soc_1204_lstm_v5k_90.0_100.0'
            # input_path = f'{base_fold}/{download_name}.png'
            # save_path = f'{save_fold_910_high}/{download_name}.png'
            # cp_file(input_path, save_path)
            num90_100_high += 1
        if json['Soc_pr_max'] >= json['soc_br_max']:
            num_80_90_max_good += 1
        if json['Soc_pr_mae'] >= json['soc_br_mae']:
            num_80_90_mae_good += 1

    elif '80.0_90.0' in json['file_info']:
        if json['soc_br_max'] >= 5.0:
            # download_name = json['file_info'].replace('_80.0_90.0','')
            # base_fold = 'D:/Code/code/submit/bms_ai/PycharmProjects/data/pic/Soc_1204_lstm_v5k_80.0_90.0'
            # input_path = f'{base_fold}/{download_name}.png'
            # save_path = f'{save_fold_890}/{download_name}.png'
            # cp_file(input_path, save_path)
            num80_90 += 1
            print(json['file_info'])
        if json['soc_br_max'] >= 3.0:
            # download_name = json['file_info'].replace('_80.0_90.0','')
            # base_fold = 'D:/Code/code/submit/bms_ai/PycharmProjects/data/pic/Soc_1204_lstm_v5k_80.0_90.0'
            # input_path = f'{base_fold}/{download_name}.png'
            # save_path = f'{save_fold_890_high}/{download_name}.png'
            # cp_file(input_path, save_path)
            num80_90_high += 1
        if json['Soc_pr_max'] >= json['soc_br_max']:
            num_90_100_max_good += 1
        if json['Soc_pr_mae'] >= json['soc_br_mae']:
            num_90_100_mae_good += 1

print('90-100:', num90_100, '\n 80-90:', num80_90)
print('90-100_high:', num90_100_high, '\n 80-90_high:', num80_90_high)

print('90-100 最大误差 Soc pre 优于 bms:', num_90_100_max_good, '\n 80-90  最大误差 Soc pre 优于 bms:',
      num_80_90_max_good)
print('90-100 mae Soc pre 优于 bms:', num_90_100_mae_good, '\n 80-90 mae Soc pre 优于 bms:', num_80_90_mae_good)

