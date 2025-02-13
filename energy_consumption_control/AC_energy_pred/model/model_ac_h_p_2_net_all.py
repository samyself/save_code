import torch
import torch.nn as nn
import torch.nn.init as init
from torch import dtype

import config_all


class MLP_phicis(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.tensor(4e2))
        self.lambda2 = nn.Parameter(torch.tensor(2e2))
        self.lambda3 = nn.Parameter(torch.tensor(5e2))
        self.lambda4 = nn.Parameter(torch.tensor(3e2))
        self.lambda5 = nn.Parameter(torch.tensor(1e1))

        self.k1 = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        effective_len = 0.137
        height = 0.0533
        width = 0.0729

        total_time = (effective_len * height * width) / (x[:, 0] / 1000)

        lambda_1 = torch.relu(self.lambda1)
        lambda_2 = torch.relu(self.lambda2)
        step_1_para_A_0_0 = 1
        step_1_para_A_0_1 = 1
        step_1_para_A_1_0 = -lambda_1 / lambda_2
        step_1_para_A_1_1 = 1

        t_b_0 = x[:, 8]  # 换热前的水温

        t_a_0 = x[:, 1]  # 排气温度
        t_a_1 = x[:, 7]  # 饱和温度

        batch_size = x.shape[0]
        # step_1_para_A = torch.tensor([[step_1_para_A_0_0, step_1_para_A_0_1], [step_1_para_A_1_0, step_1_para_A_1_1]],device=step_1_para_A_1_0.device)
        # step_1_para_A = step_1_para_A.unsqueeze(0).repeat(batch_size, 1, 1)
        # step_1_para_b = torch.cat([t_a_0.view(-1, 1), t_b_0.view(-1, 1)], dim=1)

        # step_1_solutions = torch.linalg.solve(step_1_para_A, step_1_para_b)
        # step_1_c_1 = step_1_solutions[:, 0]
        # step_1_c_2 = step_1_solutions[:, 1]
        step_1_c_1 = lambda_2 * (t_a_0 - t_b_0) / (lambda_1 + lambda_2)
        step_1_c_2 = (lambda_1 * t_a_0 + lambda_2 * t_b_0) / (lambda_1 + lambda_2)

        # 直接计算时间
        t_1 = torch.log((torch.relu((t_a_1 - step_1_c_2) / (step_1_para_A_0_0 * step_1_c_1))) + 1e-10) / (-(lambda_1 + lambda_2))
        t_1 = torch.relu(t_1)
        pred_t_a_1 = step_1_para_A_0_0 * step_1_c_1 * torch.exp(-(lambda_1 + lambda_2) * t_1) + step_1_c_2
        pred_t_b_1 = step_1_para_A_1_0 * step_1_c_1 * torch.exp(-(lambda_1 + lambda_2) * t_1) + step_1_c_2

        remain_time = torch.relu(total_time - t_1)
        '''物态变化第二阶 冷凝剂保持气液混合状态'''
        lambda_3 = torch.relu(self.lambda3)
        lambda_3_xs = lambda_3 % 1
        lambda_3_xs = torch.clamp(lambda_3_xs*1e3,min=0)
        lambda_3 = lambda_3//1 + lambda_3_xs/1e3

        # t_2 = torch.relu(self.t_2)

        # rate = torch.relu(self.k1)
        time_weight_step_2 = 0.6
        t_2 = remain_time * time_weight_step_2
        pred_t_b_2 = (pred_t_b_1 - t_a_1) * torch.exp(-lambda_3 * t_2) + t_a_1

        '''物态变化第一阶 冷凝剂由气液混合状态化为液态'''
        lambda_4 = torch.relu(self.lambda4)
        lambda_5 = torch.relu(self.lambda5)
        # t_3 = torch.relu(self.t_3)
        t_3 = remain_time * (1 - time_weight_step_2)

        # 初始的风温和制冷剂温度
        t_w_2 = pred_t_b_2
        t_r_2 = t_a_1

        step_3_para_b_0 = t_r_2
        step_3_para_b_1 = t_w_2

        step_3_para_A_0_0 = 1
        step_3_para_A_0_1 = 1
        step_3_para_A_1_0 = -lambda_4 / lambda_5
        step_3_para_A_1_1 = 1

        # step_3_para_A = torch.tensor([[step_3_para_A_0_0, step_3_para_A_0_1], [step_3_para_A_1_0, step_3_para_A_1_1]],
        #                              device=step_3_para_A_1_0.device)
        # step_3_para_A = step_3_para_A.unsqueeze(0).repeat(batch_size, 1, 1)
        # step_3_para_b = torch.cat([step_3_para_b_0.view(-1, 1), step_3_para_b_1.view(-1, 1)], dim=1)

        # step_3_solutions = torch.linalg.solve(step_3_para_A, step_3_para_b)
        # step_3_c_1 = step_3_solutions[:, 0]
        # step_3_c_2 = step_3_solutions[:, 1]
        step_3_c_1 = lambda_5 * (step_3_para_b_0 - step_3_para_b_1) / (lambda_4 + lambda_5)
        step_3_c_2 = (lambda_4 * step_3_para_b_0 + lambda_5 * step_3_para_b_1) / (lambda_4 + lambda_5)

        pred_t_a_3 = step_3_para_A_0_0 * step_3_c_1 * torch.exp(-(lambda_4 + lambda_5) * t_3) + step_3_c_2
        pred_t_b_3 = step_3_para_A_1_0 * step_3_c_1 * torch.exp(-(lambda_4 + lambda_5) * t_3) + step_3_c_2

        return pred_t_b_3, pred_t_a_3


class MLP_zi1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 1)
        self.bn0 = nn.BatchNorm1d(5)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = torch.cat((x[:, 8].unsqueeze(1), x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1), x[:, 2].unsqueeze(1), x[:, 3].unsqueeze(1)), dim=1)
        x2 = self.bn0(x1)
        x3 = self.relu(self.linear1(x2))
        x4 = self.bn1(x3)
        x5 = self.relu(self.linear2(x4))
        x6 = self.bn2(x5)
        x7 = self.relu(self.linear3(x6))
        return x7


class High_press_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 128)
        self.linear2 = nn.Linear(128, 16)
        self.linear3 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.bn0 = nn.BatchNorm1d(8)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(16)

    def forward(self, x):
        # x1 = torch.cat((x[:,0:1],x[:,1:2],x[:,2:3],x[:,3:4],x[:,5:6],x[:,7:8],x[:,8:9],x[:,9:10]),1)
        x1 = x
        x2 = self.bn0(x1)
        x3 = self.gelu(self.linear1(x2))
        x4 = self.bn1(x3)
        x5 = self.gelu(self.linear2(x4))
        x6 = self.bn2(x5)
        x7 = self.gelu(self.linear3(x6))
        return x7

def hi_pressure_temp_inter(x):
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
              [88, 3113.6], [90, 3244.2], [92, 3379.3], [94, 3519.3], [96, 3664.5]]),dtype=torch.float32)
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

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = MLP_phicis()
        self.net2 = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 1))
        self.net3 = High_press_3()

        self.r134_p = torch.tensor(config_all.r134_p_t[:, 1], dtype=torch.float)
        self.r134_t = torch.tensor(config_all.r134_p_t[:, 0], dtype=torch.float)

        # 冻结 net1 的参数
        for param in self.net1.parameters():
            param.requires_grad = False

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        hi_pressure = x[:, 7]
        hi_pressure_temp = hi_pressure_temp_inter(hi_pressure)


        x1 = torch.cat((x[:, 0:7], hi_pressure_temp.unsqueeze(1), x[:, 8:9]), dim=1)
        y1, y2 = self.net1(x1)
        x2 = torch.cat((y1.unsqueeze(1), x1[:, 4].unsqueeze(1), x1[:, 0].unsqueeze(1)), dim=1)
        out2 = self.net2(x2)
        z1 = out2[:, 0:1]
        x3 = torch.cat((x1[:, 0:1], x1[:, 1:2], x1[:, 2:3], x1[:, 3:4], z1, x1[:, 5:6], x1[:, 6:7], x1[:, 7:8]), 1)
        out3 = self.net3(x3)
        return torch.cat((y1.unsqueeze(1), z1, out3[:, 0].unsqueeze(1), out3[:, 1].unsqueeze(1)),dim=1)[0,:]


def inter1D(x_list, y_list, x):
    if not isinstance(x_list, torch.Tensor):
        x_list = torch.tensor(x_list)
    if not isinstance(y_list, torch.Tensor):
        y_list = torch.tensor(y_list)
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

    y = y_list[x_index] + (x - x_list[x_index]) * (y_list[x_index + 1] - y_list[x_index]) / (x_list[x_index + 1] - x_list[x_index])
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
