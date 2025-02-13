import torch
import torch.nn as nn
import torch.nn.init as init

from torch.ao.nn.quantized.functional import clamp

from model.data_utils_common import inter2D, inter1D, searchsorted

# 温度vs饱和压力转换
def tem_sat_press_func(val):
    # 输入压力
    # 输出tem：摄氏度
    device = val.device
    # 制冷剂温度vs饱和压力
    tem_sat_press = torch.tensor([[-62.0, 13.9], [-60.0, 15.9], [-58.0, 18.1], [-56.0, 20.5], [-54.0, 23.2],
                                  [-52.0, 26.2], [-50.0, 29.5], [-48.0, 33.1], [-46.0, 37.0], [-44.0, 41.3],
                                  [-42.0, 46.1], [-40.0, 51.2], [-38.0, 56.8], [-36.0, 62.9], [-34.0, 69.5],
                                  [-32.0, 76.7], [-30.0, 84.4], [-28.0, 92.7], [-26.0, 101.7], [-24.0, 111.3],
                                  [-22.0, 121.6], [-20.0, 132.7], [-18.0, 144.6], [-16.0, 157.3], [-14.0, 170.8],
                                  [-12.0, 185.2], [-10.0, 200.6], [-8.0, 216.9], [-6.0, 234.3], [-4.0, 252.7],
                                  [-2.0, 272.2], [0.0, 292.8], [2.0, 314.6], [4.0, 337.7], [6.0, 362.0],
                                  [8.0, 387.6], [10.0, 414.6], [12.0, 443.0], [14.0, 472.9], [16.0, 504.3],
                                  [18.0, 537.2], [20.0, 571.7], [22.0, 607.9], [24.0, 645.8], [26.0, 685.4],
                                  [28.0, 726.9], [30.0, 770.2], [32.0, 815.4], [34.0, 862.6], [36.0, 911.8],
                                  [38.0, 963.2], [40.0, 1016.6], [42.0, 1072.2], [44.0, 1130.1], [46.0, 1190.3],
                                  [48.0, 1252.9], [50.0, 1317.9], [52.0, 1385.4], [54.0, 1455.5], [56.0, 1528.2],
                                  [58.0, 1603.6], [60.0, 1681.8], [62.0, 1762.8], [64.0, 1846.7], [66.0, 1933.7],
                                  [68.0, 2023.7], [70.0, 2116.8], [72.0, 2213.2], [74.0, 2313.0], [76.0, 2416.1],
                                  [78.0, 2522.8], [80.0, 2633.2], [82.0, 2747.3], [84.0, 2865.3], [86.0, 2987.4],
                                  [88.0, 3113.6], [90.0, 3244.2], [92.0, 3379.3], [94.0, 3519.3], [96.0, 3664.5]],dtype=torch.float).to(device)
    # 将输入的压力转换为 PyTorch 张量
    if isinstance(val, list):
        val = torch.tensor(val)
    # 确保输入张量是连续的
    val = val.view(-1,1).contiguous()

    press_list = tem_sat_press[:, 1].unsqueeze(1)
    temp_list = tem_sat_press[:, 0].unsqueeze(1)
    output= inter1D(press_list, temp_list, val)
    # x_index = x_index.to(dtype=torch.float)
    # output = output[:,0]
    return output
# def tem_sat_press(press=None, tem=None):
#     # 输出tem：摄氏度
#     # 输出prss：MPa
#     if press is not None and tem is None:
#         mode = 1
#         val = press
#     elif tem is not None and press is None:
#         mode = 0
#         val = tem
#     else:
#         print("error")
#         return None
#     # 制冷剂温度vs饱和压力
#     tem_sat_press = torch.tensor([[-62.0, 13.9], [-60.0, 15.9], [-58.0, 18.1], [-56.0, 20.5], [-54.0, 23.2],
#                                   [-52.0, 26.2], [-50.0, 29.5], [-48, 33.1], [-46, 37.0], [-44, 41.3],
#                                   [-42, 46.1], [-40, 51.2], [-38, 56.8], [-36, 62.9], [-34, 69.5],
#                                   [-32, 76.7], [-30, 84.4], [-28, 92.7], [-26, 101.7], [-24, 111.3],
#                                   [-22, 121.6], [-20, 132.7], [-18, 144.6], [-16, 157.3], [-14, 170.8],
#                                   [-12, 185.2], [-10, 200.6], [-8, 216.9], [-6, 234.3], [-4, 252.7],
#                                   [-2, 272.2], [0, 292.8], [2, 314.6], [4, 337.7], [6, 362.0],
#                                   [8, 387.6], [10, 414.6], [12, 443.0], [14, 472.9], [16, 504.3],
#                                   [18, 537.2], [20, 571.7], [22, 607.9], [24, 645.8], [26, 685.4],
#                                   [28, 726.9], [30, 770.2], [32, 815.4], [34, 862.6], [36, 911.8],
#                                   [38, 963.2], [40, 1016.6], [42, 1072.2], [44, 1130.1], [46, 1190.3],
#                                   [48, 1252.9], [50, 1317.9], [52, 1385.4], [54, 1455.5], [56, 1528.2],
#                                   [58, 1603.6], [60, 1681.8], [62, 1762.8], [64, 1846.7], [66, 1933.7],
#                                   [68, 2023.7], [70, 2116.8], [72, 2213.2], [74, 2313.0], [76, 2416.1],
#                                   [78, 2522.8], [80, 2633.2], [82, 2747.3], [84, 2865.3], [86, 2987.4],
#                                   [88, 3113.6], [90, 3244.2], [92, 3379.3], [94, 3519.3], [96, 3664.5]])
#     # 将输入的压力转换为 PyTorch 张量
#     if isinstance(val, list):
#         val = torch.tensor(val)
#     # 确保输入张量是连续的
#     val = val.contiguous()
#
#     # 找到压力在表中的位置
#     val_idx = torch.searchsorted(tem_sat_press[:, mode].contiguous(), val) - 1
#     # 确保索引在有效范围内
#     val_idx = torch.clamp(val_idx, 0, tem_sat_press.shape[0] - 2)
#
#     output1 = tem_sat_press[val_idx, 0]
#
#     output2 = tem_sat_press[val_idx + 1, 0]
#
#     val_w1 = tem_sat_press[val_idx, 1]
#     val_w2 = tem_sat_press[val_idx + 1, 1]
#
#     w = (val - val_w1) / (val_w2 - val_w1)
#     output = w * (output2 - output1) + output1
#     return output


def init_weights_xavier_uniform(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)




class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, use_batchnorm=True, use_dropout=True):
        super(MLP, self).__init__()
        self.use_batchnorm = use_batchnorm
        layers = []

        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Tanh())

        # 添加额外的隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.Tanh())
            if use_dropout:
                layers.append(nn.Dropout(0.5))

        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # 创建序列模型
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp_p_h_2_max = 115.0
        self.temp_p_h_2_min = 40.0
        self.temp_p_h_5_max = 65.0
        self.temp_p_h_5_min = 20.0
        self.hi_pressure_max = 2400.0
        self.hi_pressure_min = 900.0
        self.temp_p_h_1_cab_heating_max = 31.0
        self.temp_p_h_1_cab_heating_min = -18.0
        self.lo_pressure_max = 700.0
        self.lo_pressure_min = 100.0
        self.aim_hi_pressure_max = 2400.0
        self.aim_hi_pressure_min = 900.0
        self.compressor_speed_max = 8000.0
        self.compressor_speed_min = 0
        self.cab_heating_status_act_pos_max = 100.0
        self.cab_heating_status_act_pos_min = 0.0
        self.ac_kp_rate_max = 4.0
        self.ac_kp_rate_min = 0.0

        # 插值表
        # 高压差-Kp表
        self.CP_PID_Kp_list = torch.tensor([1.5, 0.8984375, 0.55078125, 0, 0, 0.44921875, 0.6015625, 0.8984375])
        self.CP_diffpress_Kp_list = torch.tensor([-10, -5, -2, -0.75, 0.75, 2, 5, 10])
        # 高压差-Ki表
        self.CP_PID_Ki_list = torch.tensor(
            [0.0500488, 0.0200195, 0.0080566, 0.0024414, 0.0024414, 0.0080566, 0.0200195, 0.0500488])
        self.CP_diffpress_Ki_list = torch.tensor([-10, -5, -2, -0.75, 0.75, 2, 5, 10])

        # 转速初值表
        self.CP_InitValue_list = torch.tensor([30, 35, 40, 45, 50, 55, 60, 65])
        self.CP_com_sped_list = torch.tensor([0, 2000, 3000, 4000, 5000, 6000, 7000, 8000])

        self.Delta = None

        self.MLP1 = MLP(input_size=4, hidden_sizes=[256, 256], output_size=1, use_batchnorm=False, use_dropout=False)
        self.MLP2 = MLP(input_size=1, hidden_sizes=[256, 256], output_size=1, use_batchnorm=False, use_dropout=False)

        init_weights_xavier_uniform(self.MLP1)
        init_weights_xavier_uniform(self.MLP2)
        self.k1 = nn.Parameter(torch.randn([1]), requires_grad=True)

    def forward(self, x1):
        torch.autograd.set_detect_anomaly(True)
        # 压缩机排气温度、内冷温度、饱和高压、压缩机进气温度、饱和低压、目标饱和高压、目标过冷度、目标过热度
        # 压缩机转速、膨胀阀开度
        x1 = x1.view(-1, 8)
        x = x1.to(dtype=torch.float)

        # 输出限制
        # 目标过冷度
        sc_tar_mode_10 = x[:, 0]
        sc_tar_mode_10 = sc_tar_mode_10.clamp(min=-100.0, max=100.0)
        # 目标过热度
        sh_tar_mode_10 = x[:, 1]
        sh_tar_mode_10 = sh_tar_mode_10.clamp(min=-100.0, max=100.0)
        # 压缩机进气温度
        temp_p_h_1_cab_heating = x[:, 2]
        temp_p_h_1_cab_heating = temp_p_h_1_cab_heating.clamp(min=-100.0, max=100.0)
        # 内冷温度
        temp_p_h_5 = x[:, 3]
        temp_p_h_5 = temp_p_h_5.clamp(min=-100.0, max=100.0)
        # 饱和高压
        hi_pressure = x[:, 4]
        hi_pressure = hi_pressure.clamp(min=0.0, max=2400.0)
        # 饱和低压
        lo_pressure = x[:, 5]
        lo_pressure = lo_pressure.clamp(min=0.0, max=1100.0)
        # 压缩机转速
        compressor_speed_last = x[:, 6]
        # 上一时刻exv_oh_pid
        last_exv_oh_pid = x[:, 7]
        last_exv_oh_pid = last_exv_oh_pid.clamp(min=0.0, max=100.0)

        # 实际过热度  = 压缩机进气温度 - 低压饱和压力对应温度
        sh_rel_mode_10 = temp_p_h_1_cab_heating- tem_sat_press_func(lo_pressure).view(-1)
        # 实际过冷度 =  高压饱和压力对应温度 - 内冷温度
        sc_rel_mode_10 = tem_sat_press_func(hi_pressure).view(-1) - temp_p_h_5
        # 过冷度偏差
        SCRaw = (sc_rel_mode_10 - sc_tar_mode_10)
        # 过热度偏差
        SCOffset = (sh_rel_mode_10 - sh_tar_mode_10)
        # SCErr
        SCErr = (SCRaw + SCOffset)

        # """
        #     开度前向传播
        # """

        Kp = inter1D(self.CP_diffpress_Kp_list, self.CP_PID_Kp_list, SCErr).view(-1)
        Ki = inter1D(self.CP_diffpress_Ki_list, self.CP_PID_Ki_list, SCErr).view(-1)
        Kd = 0


        Ki_zero_mask = (SCErr < 0) | (SCRaw < 0) | (last_exv_oh_pid >= 38) | (SCRaw != 0)
        Ki[Ki_zero_mask] = 0

        # if self.Delta == None:
        if True:
            # self.Delta = inter1D(self.CP_com_sped_list, self.CP_InitValue_list, compressor_speed_last)
            self.Delta = 30

        self.Delta = (self.Delta + Ki * SCErr)
        offset = Kp * SCErr + self.Delta

        use_x = torch.cat((temp_p_h_5.unsqueeze(1), hi_pressure.unsqueeze(1),
                           temp_p_h_1_cab_heating.unsqueeze(1), lo_pressure.unsqueeze(1)), dim=1)

        part1 = self.MLP1(use_x)

        # part2 = self.MLP2(compressor_speed_last.unsqueeze(1))

        exv_oh_pid = last_exv_oh_pid + (offset + part1[:,0]) * self.k1
        # exv_oh_pid = offset

        exv_oh_pid = torch.clamp(exv_oh_pid,min=12.0,max=100.0)
        # exv_oh_pid = torch.relu(exv_oh_pid - 12.0) + 12.0
        # exv_oh_pid = 100.0 - torch.relu(100.0 - exv_oh_pid)

        return exv_oh_pid.view(-1, 1)

import random

if __name__ == '__main__':
    model = MyModel()
    model.eval()
    # random.seed(42)
    # matrix = [[round(random.uniform(-10000, 10000),4) for _ in range(8)] for _ in range(10000)]
    # matrix = torch.tensor(matrix, dtype=torch.float32)
    matrix = torch.ones(1, 8) * -100000
    y = model(matrix)
    if y.isnan().any():
        print("存在NaN值")
    else:
        print("不存在NaN值")
    print(y.shape)
