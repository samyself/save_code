import os

import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Union

import sys

# project_folder = os.path.abspath('..')
# sys.path.append(os.path.join(project_folder, 'common'))
from AC_energy_pred.model.data_utils_common import inter2D, inter1D, searchsorted





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


def init_weights_xavier_uniform(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)


class MyCabPosModel(nn.Module):
    def __init__(self, input_size=6, output_size=1, hidden_size1=None, hidden_size2=None, dropout=0.2):
        super(MyCabPosModel, self).__init__()
        # 开度建模
        self.cab_pos_part1 = MLP(input_size=1, hidden_sizes=[32, 32], output_size=1, use_batchnorm=True)
        # self.cab_pos_part2 = MLP(input_size=1, hidden_sizes=[32, 32], output_size=1, use_batchnorm=True)
        self.cab_pos_part3 = MLP(input_size=5, hidden_sizes=[32, 32], output_size=1, use_batchnorm=True)
        init_weights_xavier_uniform(self.cab_pos_part1)
        # init_weights_xavier_uniform(self.cab_pos_part2)
        init_weights_xavier_uniform(self.cab_pos_part3)

    def forward(self, SCErr,exv_pid_iout, use_x):
        part_1_input = SCErr.unsqueeze(1)
        part_2_input = exv_pid_iout.unsqueeze(1)
        part1 = self.cab_pos_part1(part_1_input)
        part2 = part_2_input
        part3 = self.cab_pos_part3(use_x)
        # cab_heating_status_act_pos = part3
        cab_heating_status_act_pos = part1  + part2 + part3
        return cab_heating_status_act_pos


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, use_batchnorm=False):
        super(MLP, self).__init__()
        self.use_batchnorm = use_batchnorm
        # use_batchnorm = False
        layers = []
        # self.net1 = nn.Linear(input_size, output_size)

        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.GELU())

        # 添加额外的隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.GELU())

        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # 创建序列模型
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # outputs = []
        outputs = self.network(x)
        # for i, layer in enumerate(self.network):
        #     x = layer(x)
        #     # if (torch.abs(x) > 65535).any():
        #     #     x = torch.clamp(x, min=-65504, max=65504)
        #     # x = torch.round(x * 1000) / 1000
        #     # x = torch.clamp(x*1e3, min=0)
        #     # x = x/1e3
        #     outputs.append(x)
        return outputs


def norm(tensor_input,max_data: Union[torch.Tensor], min_data: Union[torch.Tensor]):
    tensor = tensor_input.view(-1,1)
    # if not isinstance(max, torch.Tensor):
    max_data_full = torch.ones((tensor.shape[0],1))*max_data
    # if not isinstance(min, torch.Tensor):
    min_data_full = torch.ones((tensor.shape[0],1))*min_data

    output_up_in1 = torch.cat((tensor, min_data_full), dim=1)
    output_up_in2 = torch.tensor([1.0, -1.0],device=tensor.device).view(2,1)
    output_up = torch.mm(output_up_in1 , output_up_in2)

    output_down_in1 = torch.cat((tensor, max_data_full, min_data_full), dim=1)
    output_down_in2 = torch.tensor([0.0, 1.0, -1.0],device=tensor.device).view(3,1)
    # output_down_in1 = torch.cat(( max_data_full, min_data_full), dim=1)
    # output_down_in2 = torch.tensor([1.0, -1.0],device=tensor.device).view(2,1)
    output_down = torch.mm(output_down_in1 , output_down_in2)

    output = output_up / output_down
    return output[:,0]

def my_relu(input_tensor):
    # 创建一个与输入张量形状相同的全零张量
    zero_tensor = torch.zeros_like(input_tensor)
    # 应用 max 函数，比较每个元素并返回较大的那个
    return torch.max(zero_tensor, input_tensor)

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

        self.dif_hi_pressure_max = 2000.0
        self.dif_hi_pressure_min = -2000.0
        self.dif_lo_pressure_max = 2000.0
        self.dif_lo_pressure_min = -2000.0

        self.cab_pos_model = MyCabPosModel(input_size=5, output_size=1)

    def forward(self, x1):
        # 压缩机排气温度、内冷温度、饱和高压、压缩机进气温度、饱和低压、目标饱和高压、目标过冷度、目标过热度
        # 压缩机转速、膨胀阀开度
        x1 = x1.view(-1,11)
        x = x1.to(dtype=torch.float)

        # x = torch.clamp(x, min=-10000.0, max=10000.0)
        device = x.device

        # 输出限制
        # 上一时刻ac_pid_pout
        ac_pid_out_hp = x[:, 0]
        ac_pid_out_hp = ac_pid_out_hp.clamp(min=0.0, max=8000.0)
        # 上一时刻饱和低压
        lo_pressure = x[:, 1]
        lo_pressure = lo_pressure.clamp(min=0.0, max=1100.0)
        # 上一时刻饱和高压
        hi_pressure = x[:, 2]
        hi_pressure = hi_pressure.clamp(min=0.0, max=2400.0)
        # 目标饱和低压
        aim_lo_pressure = x[:, 3]
        aim_lo_pressure = aim_lo_pressure.clamp(min=0.0, max=1100.0)
        # 目标饱和高压
        aim_hi_pressure = x[:, 4]
        aim_hi_pressure = aim_hi_pressure.clamp(min=0.0, max=2400.0)
        # 上一时刻EXV_Oh_PID_Iout
        exv_pid_iout = x[:, 5]
        exv_pid_iout = exv_pid_iout.clamp(min=0.0, max=100.0)
        # 目标过冷度
        sc_tar_mode_10 = x[:, 6]
        sc_tar_mode_10 = sc_tar_mode_10.clamp(min=-100.0, max=100.0)
        # 目标过热度
        sh_tar_mode_10 = x[:, 7]
        sh_tar_mode_10 = sh_tar_mode_10.clamp(min=-100.0, max=100.0)
        # 压缩机排气温度
        temp_p_h_2 = x[:, 8]
        temp_p_h_2 = temp_p_h_2.clamp(min=-100.0, max=200.0)
        # 内冷温度
        temp_p_h_5 = x[:, 9]
        temp_p_h_5 = temp_p_h_5.clamp(min=-100.0, max=100.0)
        # 压缩机进气温度
        temp_p_h_1_cab_heating = x[:, 10]
        temp_p_h_1_cab_heating = temp_p_h_1_cab_heating.clamp(min=-100.0, max=100.0)

        # 实际过热度  = 压缩机排气温度 - 低压饱和压力对应温度
        # low_temp = tem_sat_press_func(lo_pressure)
        low_temp=torch.zeros(lo_pressure.shape).view(-1,1)

        sh_rel_mode_10 = temp_p_h_2 - low_temp[:,0]
        # 实际过冷度 =  高压饱和压力对应温度 - 内冷温度
        hi_temp = tem_sat_press_func(hi_pressure)
        # hi_temp = torch.zeros(lo_pressure.shape).view(-1,1)
        sc_rel_mode_10 = hi_temp[:,0] - temp_p_h_5

        # SCErr 过热度偏差 过冷度偏差
        SCERR_in1 = torch.cat((sh_rel_mode_10.view(-1,1),sh_tar_mode_10.view(-1,1),
                               sc_rel_mode_10.view(-1,1),sc_tar_mode_10.view(-1,1)),dim=1)
        SCERR_in2 = torch.tensor([1.0, -1.0, 1.0, -1.0],device=device).view(4,1)
        SCErr = torch.mm(SCERR_in1, SCERR_in2)[:,0]
        # SCErr = torch.ones_like(hi_pressure).to(torch.float)

        # 计算最小理论压缩机转速

        """
            压缩机转速前向传播
        """

        compressor_speed = ac_pid_out_hp.view(-1,1)

        compressor_speed = torch.clamp(compressor_speed, min=800.0)

        """
            膨胀阀开度前向传播
        """

        # 排气 压缩机温度差 内冷 饱高 压缩比 包和高压差
        use_x = torch.cat((temp_p_h_2.view(-1,1), temp_p_h_5.view(-1,1), hi_pressure.view(-1,1),
                           temp_p_h_1_cab_heating.view(-1,1), lo_pressure.view(-1,1)), dim=1)
        cab_heating_status_act_pos = self.cab_pos_model(SCErr, exv_pid_iout, use_x)

        cab_heating_status_act_pos = torch.clamp(cab_heating_status_act_pos, min=0.0, max=100.0)

        output = torch.cat((compressor_speed, cab_heating_status_act_pos), dim=1)
        return output

if __name__ == '__main__':
    import random
    model = MyModel()
    model.eval()
    # random.seed(42)
    matrix = [[round(random.uniform(-10000, 10000),4) for _ in range(11)] for _ in range(10000)]
    matrix = torch.tensor(matrix, dtype=torch.float32)
    # matrix = torch.zeros((1, 11))
    y = model(matrix)
    if y.isnan().any():
        print("存在NaN值")
    else:
        print("不存在NaN值")
    print(y.shape)