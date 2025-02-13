import argparse
import os.path
import pickle

import torch
from pyexpat import features
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from model.inter_func_hpt import inter1
from draw_pic import draw_pred

list = torch.tensor(([[-62, 13.9], [-60, 15.9], [-58, 18.1], [-56, 20.5], [-54, 23.2],
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
                      [88, 3113.6], [90, 3244.2], [92, 3379.3], [94, 3519.3], [96, 3664.5]]), dtype=torch.float32)



def hi_pressure_temp_inter(x):

    x_list = list[:,1]
    y_list = list[:,0]
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    # 直接将x中的最大值与最小值作为x_list的最小、大边界
    x_min = torch.min(x)
    x_max = torch.max(x)
    if x_min < x_list[0]:
        y_left = (y_list[1] - y_list[0]) / (x_list[1] - x_list[0]) * (x_min - x_list[0]) + y_list[0]
        y_list[0] = y_left
        x_list[0] = x_min
    if x_max > x_list[-1]:
        y_right = (y_list[-1] - y_list[-2]) / (x_list[-1] - x_list[-2]) * (x_max - x_list[-1]) + y_list[-1]
        y_list[-1] = y_right
        x_list[-1] = x_max

        # 确保输入张量是连续的
    x = x.contiguous()

    # 找到输入低压和高压在表中的位置
    x_index = searchsorted(x_list, x) - 1

    y = y_list[x_index] + (x - x_list[x_index]) * (y_list[x_index + 1] - y_list[x_index]) / (
                x_list[x_index + 1] - x_list[x_index])
    return y

def searchsorted(sorted_sequence, values, out_int32: bool = False, right: bool = False) -> torch.LongTensor:
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
    if len(values.shape) == 2:
        values = values[:, 0]

    indices = torch.zeros_like(values)
    for i in range(values.shape[0]):
        left, right_bound = 0, len(sorted_sequence)
        value = values[i]
        while left < right_bound:
            mid = (left + right_bound) // 2
            if (sorted_sequence[mid] < value) or (right and sorted_sequence[mid] <= value):
                left = mid + 1
            else:
                right_bound = mid

        indices[i] = left

    indices = indices.to(torch.int32 if out_int32 else torch.int64)
    return indices





#训练
def train(model, optimizer, criterion, train_loader, epoch):
    model.train()
    train_loss = []
    for idx in tqdm(range(epoch)):
        loss_list = []
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        train_loss.append(sum(loss_list)/len(loss_list))
        print(f'Epoch [{idx + 1}/{epoch}], Loss: {train_loss[-1]}')
    return train_loss

def  test(model, criterion, test_loader):
    model.eval()
    test_loss = []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
    return test_loss


if  __name__ == '__main__':
    # 定义模型、优化器、损失函数
    model = inter1()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    train_x_list = list[:, 1]
    train_y_list = list[:, 0]

    # 可视化
    series_list = [
        [train_x_list, train_y_list],
    ]
    series_name_list = [
        # ['pred_refrigerant_mix_temp', 'refrigerant_mix_temp'],
        ['train_loss', 'test_loss'],
    ]
    pic_name = 'inter_hp_t1' + '.png'
    result_pic_folder = './data/train_result/pic/inter1'
    # print(series_list)
    draw_pred(series_list, series_name_list, result_pic_folder, pic_name)



    test_x_list = train_x_list + torch.randn(train_x_list.shape) * 10
    test_y_list = hi_pressure_temp_inter(test_x_list)
    trian_loaer = DataLoader(torch.utils.data.TensorDataset(train_x_list, train_y_list), batch_size=8, shuffle=True)
    test_loader = DataLoader(torch.utils.data.TensorDataset(test_x_list, test_y_list), batch_size=8, shuffle=True)

    train_loss = train(model, optimizer, criterion, trian_loaer, epoch=100)
    test_loss = test(model, criterion, test_loader)

    # 可视化
    series_list = [
        [train_loss, test_loss],
    ]
    series_name_list = [
        # ['pred_refrigerant_mix_temp', 'refrigerant_mix_temp'],
        ['train_loss', 'test_loss'],
    ]
    pic_name = 'inter_hp_t1' + '.png'
    result_pic_folder = './data/train_result/pic/inter1'
    # print(series_list)
    draw_pred(series_list, series_name_list, result_pic_folder, pic_name)