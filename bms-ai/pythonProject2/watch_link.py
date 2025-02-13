import os
from cProfile import label

import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import matplotlib
import numpy as np
import torch.nn as nn
from itertools import cycle

matplotlib.rc("font",family='YouYuan')

def searchsorted(sorted_sequence, values, out_int32=False, right=False):
    """
    手动实现 torch.searchsorted 功能。

    参数:
        sorted_sequence (Tensor): 一个有序的一维张量。
        values (Tensor or Scalar): 要查找插入位置的值。
        out_int32 (bool, optional): 输出索引的类型是否为 int32，默认为 False（int64）。
        right (bool, optional): 是否使用右侧边界，默认为 False。

    返回:
        Tensor: 插入位置的索引。
    """
    if not isinstance(values, torch.Tensor):
        values = torch.tensor([values])

    indices = []
    for value in values:
        left, right_bound = 0, len(sorted_sequence)

        while left < right_bound:
            mid = (left + right_bound) // 2
            if (sorted_sequence[mid] < value) or (right and sorted_sequence[mid] <= value):
                left = mid + 1
            else:
                right_bound = mid

        indices.append(left)

    indices_tensor = torch.tensor(indices, dtype=torch.int32 if out_int32 else torch.int64)
    return indices_tensor

# 温度vs饱和压力转换
def tem_sat_press(press=None,tem=None):
    # 输出tem：摄氏度
    # 输出prss：MPa
    if press is not None and tem is None:
       mode = 1
       val = press
    elif tem is not None and press is None:
       mode = 0
       val = tem
    else:
       print("error")
       return None
    # 制冷剂温度vs饱和压力
    tem_sat_press = torch.tensor([[-62, 13.9], [-60, 15.9], [-58, 18.1], [-56, 20.5], [-54, 23.2],
                     [-52, 26.2], [-50, 29.5], [-48, 33.1], [-46, 37.0], [-44, 41.3],
                     [-42, 46.1], [-40, 51.2], [-38, 56.8], [-36, 62.9], [-34, 69.5],
                     [-32, 76.7], [-30, 84.4], [-28, 92.7], [-26, 101.7], [-24, 111.3],
                     [-22, 121.6], [-20, 132.7], [-18, 144.6], [-16, 157.3], [-14, 170.8],
                     [-12, 185.2], [-10, 200.6], [-8, 216.9], [-6, 234.3], [-4, 252.7],
                     [-2, 272.2], [0, 292.8], [2, 314.6], [4, 337.7], [6, 362.0],
                     [8, 387.6], [10, 414.6], [12, 443.0], [14, 472.9], [16, 504.3],
                     [18, 537.2], [20, 571.7], [22, 607.9], [24, 645.8], [26, 685.4],
                     [28, 726.9], [30, 770.2], [32, 815.4], [34, 862.6], [36, 911.8],
                     [38, 963.2], [40, 1016.6], [42, 1072.2], [44, 1130.1], [46, 1190.3],
                     [48, 1252.9], [50, 1317.9], [52, 1385.4], [54, 1455.5], [56, 1528.2],
                     [58, 1603.6], [60, 1681.8], [62, 1762.8], [64, 1846.7], [66, 1933.7],
                     [68, 2023.7], [70, 2116.8], [72, 2213.2], [74, 2313.0], [76, 2416.1],
                     [78, 2522.8], [80, 2633.2], [82, 2747.3], [84, 2865.3], [86, 2987.4],
                     [88, 3113.6], [90, 3244.2], [92, 3379.3], [94, 3519.3], [96, 3664.5]])
    # 将输入的压力转换为 PyTorch 张量
    if isinstance(val, list):
        val = torch.tensor(val)
    # 确保输入张量是连续的
    val = val.contiguous()

    # 找到压力在表中的位置
    val_idx = searchsorted(tem_sat_press[:,mode], val) - 1
    # 确保索引在有效范围内
    val_idx = torch.clamp(val_idx, 0,  tem_sat_press.shape[0]- 2)

    def mode_reverse(mode):
        if mode == 0:
            return 1
        elif mode == 1:
            return 0
        else:
            print("mode error")
            return None

    output1 = tem_sat_press[val_idx, mode_reverse(mode)]

    output2 = tem_sat_press[val_idx+1, mode_reverse(mode)]

    val_w1 = tem_sat_press[val_idx, mode]
    val_w2 = tem_sat_press[val_idx+1, mode]


    w = (val - val_w1) / (val_w2 - val_w1)
    output = w * (output2 - output1) + output1
    return output



def load_csv(file_path):
    df=pd.read_csv(file_path)
    data=df.to_dict('records')
    return data


def file_data_in(file_path,file_name_list):
    input_data = []
    if file_name_list == 'all':
        file_name_list = os.listdir(file_path)
        # file_name_list = [os.path.join(file_path, file_name) for file_name in all_file_name]

    for file_name in file_name_list:
        input_data_dir = os.path.join(file_path, file_name)
        input_data.extend(load_csv(input_data_dir))
    return input_data




def pic_look(data, title, names, file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Define color and line style cycles
    colors = plt.cm.tab10.colors  # Use a color map with high contrast
    line_styles = ['-', '--', '-.', ':']  # Different line styles for added clarity
    color_cycle = cycle(colors)
    line_style_cycle = cycle(line_styles)

    # Create a new figure
    plt.figure(figsize=(15, 6))

    for i, row in enumerate(data):
        # Get the next color and line style
        color = next(color_cycle)
        line_style = next(line_style_cycle)

        # Plot each row with unique color and line style
        plt.plot(row, label=names[i], color=color, linestyle=line_style)

    # Set title and labels
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save and close the figure
    plt.savefig(f'{file_path}/{title}.png')
    plt.show()
    plt.close()


def norm(tensor):

    tensor_max = torch.max(tensor,dim=0)
    tensor_min = torch.min(tensor, dim=0)
    tensor = (tensor - tensor_min.values)/(tensor_max.values - tensor_min.values+1e-6)

    return tensor

if __name__ == '__main__':
    file_folder = '../data/filter_data/data_handle_BmsSoc_15_filter_v1'
    save_path = '../data/data_anyls/1000-SOC-filter-badcase'
    all_file_name = os.listdir(file_folder)
    xgx1_data = []
    xgx2_data = []
    xgx3_data = []
    xgx4_data = []


    # for name_index in tqdm(range(len(all_file_name))):
    for name_index in tqdm(range(100)):

        file_name = all_file_name[name_index]
        file_path = os.path.join(file_folder, file_name)
        input_data = file_data_in(file_folder, file_name_list=[file_name])
        input_list = []
        for row in input_data:
            input_list.append(
                [row['Volt'], row['Temp'],
                 row['Curr'],row['Ah'],row['avg_current60'],row['avg_voltage60'],
                 row['avg_current300'], row['avg_voltage300'],
                 row['avg_current600'], row['avg_voltage600'],
                 row['avg_current_all'], row['avg_voltage_all'],
                 row['BmsSoc'],row['TrueSoc']])
        input_list = torch.tensor(input_list)

        input_list =norm(input_list)
        A0 = input_list[:, 0]
        A1 = input_list[:, 1]

        A2 = input_list[:, 2]
        A3 = input_list[:, 3]
        A4 = input_list[:, 4]
        A5 = input_list[:, 5]
        A6 = input_list[:, 6]
        A7 = input_list[:, 7]

        A8 = input_list[:, 8]
        A9 = input_list[:, 9]
        A10 = input_list[:, 10]
        A11 = input_list[:, 11]
        A12 = input_list[:, 12]
        B = input_list[:, 13]

        #
        pic_list = [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,B]
        # pic_list = [ A11,A12, B]
        name_list = ['Volt','Temp','Curr','Ah','avg_current60','avg_voltage60','avg_current300','avg_voltage300',
                     'avg_current600','avg_voltage600','avg_current_all','avg_voltage_all','BmsSoc','Soc_real']
        # name_list = ['voltage_all', 'sco', 'true']

        file_name = file_name.replace('.cvs','')
        pic_look(pic_list,file_name,name_list,save_path)



