import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.onnx.symbolic_opset11 import unsqueeze

from AC_energy_pred import config
# from ..data_utils import cal_h

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
    val_idx = torch.searchsorted(tem_sat_press[:,mode].contiguous(), val) - 1
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


# 压缩机转速下限
def com_speed_min_bilinear_interpolation(lo_press, high_press):  # 双线性插值
    # 插值表
    lo_press_table = torch.tensor([100.0, 150, 200, 250, 300, 350, 400, 450, 500, 550])
    high_press_table = torch.tensor([200.0, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])

    # 第二个维度(高压压力)：200,400,600,800,1000,1200,1400,1600,1800,2000                     第一个维度(低压压力)
    com_speed_min_table = torch.tensor([[2000.0, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000],  # 100
                                        [1600, 1600, 1600, 1600, 1600, 1700, 1800, 1900, 2000, 2000],  # 150
                                        [1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1600, 2000],  # 200
                                        [900, 900, 950, 1000, 1050, 1100, 1150, 1200, 1600, 2000],  # 250
                                        [800, 800, 800, 800, 900, 1000, 1100, 1200, 1600, 2000],  # 300
                                        [800, 800, 800, 800, 800, 900, 1050, 1200, 1600, 2000],  # 350
                                        [800, 800, 800, 800, 800, 800, 1000, 1200, 1600, 2000],  # 400
                                        [800, 800, 800, 800, 800, 800, 950, 1200, 1600, 2000],  # 450
                                        [800, 800, 800, 800, 800, 800, 900, 1200, 1600, 2000],  # 500
                                        [800, 800, 800, 800, 800, 800, 850, 1200, 1600, 2000]])  # 550

    # 将输入的压力转换为 PyTorch 张量
    if isinstance(lo_press, list):
        lo_press = torch.tensor(lo_press)
    if isinstance(high_press, list):
        high_press = torch.tensor(high_press)

        # 确保输入张量是连续的
    lo_press = lo_press.contiguous()
    high_press = high_press.contiguous()

    # 找到输入低压和高压在表中的位置
    lo_press_idx = torch.searchsorted(lo_press_table, lo_press) - 1
    high_press_idx = torch.searchsorted(high_press_table, high_press) - 1

    # 确保索引在有效范围内
    lo_press_idx = torch.clamp(lo_press_idx, 0, len(lo_press_table) - 2)
    high_press_idx = torch.clamp(high_press_idx, 0, len(high_press_table) - 2)

    # 获取四个最近的点
    Q11 = com_speed_min_table[lo_press_idx, high_press_idx]
    Q12 = com_speed_min_table[lo_press_idx, high_press_idx + 1]
    Q21 = com_speed_min_table[lo_press_idx + 1, high_press_idx]
    Q22 = com_speed_min_table[lo_press_idx + 1, high_press_idx + 1]

    # 计算 x 和 y 方向的比例
    x_ratio = (lo_press - lo_press_table[lo_press_idx]) / (
            lo_press_table[lo_press_idx + 1] - lo_press_table[lo_press_idx])

    y_ratio = (high_press - high_press_table[high_press_idx]) / (
            high_press_table[high_press_idx + 1] - high_press_table[high_press_idx])

    # 在 x 方向上进行线性插值
    R1 = x_ratio * (Q21 - Q11) + Q11
    R2 = x_ratio * (Q22 - Q12) + Q12

    # 在 y 方向上进行线性插值
    P = y_ratio * (R2 - R1) + R1
    return P

def init_weights_xavier_uniform(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

class MyCabPosModel(nn.Module):
    def __init__(self, input_size=12,output_size=1, num_channels=None, kernel_size = 3, hidden_size=None, dropout = 0.5):
        super(MyCabPosModel,self).__init__()
        if hidden_size is None:
            hidden_size1 = [256, 256]

        # 开度建模
        self.Cd = nn.Parameter(torch.rand(1), requires_grad=True)
        self.A0 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.rowL = nn.Parameter(torch.rand(1), requires_grad=True)
        self.K = nn.Parameter(torch.rand(1), requires_grad=True)
        self.n = nn.Parameter(torch.rand(1), requires_grad=True)

        # 控制系数bias
        layers = []
        for i in range(len(hidden_size)):
            in_channels = 12 if i == 0 else hidden_size[i-1]
            out_channels = hidden_size[i]
            layers += [nn.Linear(in_channels, out_channels),nn.ReLU(),nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_size[-1], output_size)]
        self.bias = nn.Sequential(*layers)

    def forward(self,diff_press,use_x2):
        mask = diff_press < 0
        diff_press = torch.where(mask, 0, diff_press)

        # 计算质量流量
        mvar = self.Cd * self.A0 * torch.sqrt(self.rowL * (diff_press))
        cab_heating_status_act_pos = self.K * (mvar ** self.n) + self.bias(use_x2)
        return cab_heating_status_act_pos[:,0]

class MyComSpeedModel(nn.Module):
    def __init__(self, input_size=6, output_size=1, hidden_size1=None, hidden_size2=None, dropout = 0.5):
        super(MyComSpeedModel,self).__init__()
        # 压缩机建模
        # 输入层
        if hidden_size1 is None:
            hidden_size1 = [256, 256]
        if hidden_size2 is None:
            hidden_size2 = [512, 512, 512]
        # 控制系数k
        layers = []
        for i in range(len(hidden_size1)):
            in_channels = 2 if i == 0 else hidden_size1[i-1]
            out_channels = hidden_size1[i]
            layers += [nn.Linear(in_channels, out_channels),nn.ReLU(),nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_size1[-1], output_size)]
        self.com_speed_k = nn.Sequential(*layers)

        # 转速偏差b
        layers = []
        for i in range(len(hidden_size2)):
            in_channels = input_size if i == 0 else hidden_size2[i-1]
            out_channels = hidden_size2[i]
            layers += [nn.Linear(in_channels, out_channels),nn.ReLU(),nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_size2[-1], output_size)]
        self.com_speed_b = nn.Sequential(*layers)
        init_weights_xavier_uniform(self.com_speed_k)
        init_weights_xavier_uniform(self.com_speed_b)

    def forward(self, aim_hi_pressure, hi_pressure, use_x1):
        input = torch.cat((aim_hi_pressure.unsqueeze(1), hi_pressure.unsqueeze(1)), dim=1)
        compressor_speed = self.com_speed_k(input) + self.com_speed_b(use_x1)
        return compressor_speed[:,0]




def norm(tensor,max,min):
    output = (tensor - min)/(max - min)
    return output

def renorm(tensor,max,min):
    output = tensor*(max - min) + min
    return output

class MLPModel(nn.Module):
    def __init__(self, input_size, output_dim, num_channels,kernel_size, dropout=0.5):
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

        self.compressor_speed_model = MyComSpeedModel()

        self.cab_pos_model = MyCabPosModel()


    def forward(self, x):
        # 压缩机排气温度、内冷温度、饱和高压、压缩机进气温度、饱和低压、目标饱和高压、目标过冷度、目标过热度
        # 压缩机转速、膨胀阀开度
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        # 输出限制
        # 压缩机排气温度
        temp_p_h_2 = x[:,0]
        # 内冷温度
        temp_p_h_5 = x[:, 1]
        # 饱和高压
        hi_pressure = x[:,2]
        # 压缩机进气温度
        temp_p_h_1_cab_heating = x[:,3]
        # 饱和低压
        lo_pressure = x[:,4]
        # 目标饱和高压
        aim_hi_pressure = x[:,5]
        # 目标饱和低压
        aim_lo_pressure = torch.clamp(x[:,6],max=700.0,min=100.0)
        # 目标过冷度
        sc_tar_mode_10 = x[:,7]
        # 目标过热度
        sh_tar_mode_10 = x[:,8]


        # 输入解耦
        # 压缩机温度差 = 压缩机排气温度 - 压缩机进气温度
        dif_temp_p_h = temp_p_h_2 - temp_p_h_1_cab_heating
        # 饱和高压差 = 目标饱和高压 - 饱和高压
        dif_hi_pressure = aim_hi_pressure - hi_pressure
        # 压缩比 = 饱和高压 / 饱和低压
        rate_pressure = hi_pressure / lo_pressure
        # 压缩差 = 饱和高压 - 饱和低压
        dif_pressure = hi_pressure - lo_pressure
        # 计算最小理论压缩机转速
        com_speed_min = com_speed_min_bilinear_interpolation(lo_pressure, hi_pressure)
        # 实际过热度  = 压缩机排气温度 - 低压饱和压力对应温度
        sh_rel_mode_10 = temp_p_h_2 - tem_sat_press(lo_pressure)
        # 实际过冷度 =  高压饱和压力对应温度 - 内冷温度
        sc_rel_mode_10 = tem_sat_press(hi_pressure) - temp_p_h_5




        com_speed_zeros_mask = hi_pressure - lo_pressure < 10

        # 归一化
        temp_p_h_2 = norm(temp_p_h_2,self.temp_p_h_2_max,self.temp_p_h_2_min)
        temp_p_h_5 = norm(temp_p_h_5,self.temp_p_h_5_max,self.temp_p_h_5_min)
        hi_pressure = norm(hi_pressure,self.hi_pressure_max,self.hi_pressure_min)
        temp_p_h_1_cab_heating = norm(temp_p_h_1_cab_heating,self.temp_p_h_1_cab_heating_max,self.temp_p_h_1_cab_heating_min)
        lo_pressure = norm(lo_pressure,self.lo_pressure_max,self.lo_pressure_min)
        aim_hi_pressure = norm(aim_hi_pressure,self.aim_hi_pressure_max,self.aim_hi_pressure_min)

        """
            压缩机转速前向传播
        """
        use_x1 = torch.cat((temp_p_h_2.unsqueeze(1), temp_p_h_5.unsqueeze(1), hi_pressure.unsqueeze(1),
                            temp_p_h_1_cab_heating.unsqueeze(1), lo_pressure.unsqueeze(1), aim_hi_pressure.unsqueeze(1)), dim=1)

        compressor_speed = self.compressor_speed_model(aim_hi_pressure, hi_pressure, use_x1)

        """
            膨胀阀开度前向传播
        """
        # 目标过热度 实际过热度 过热度差值 目标过冷度 实际过冷度 过冷度差值 排气 压缩机温度差 内冷 饱高 压缩比 饱和高压差
        use_x2 = torch.cat((sh_tar_mode_10.unsqueeze(1), sh_rel_mode_10.unsqueeze(1),(sh_tar_mode_10 - sh_rel_mode_10).unsqueeze(1),
                            sc_tar_mode_10.unsqueeze(1), sc_rel_mode_10.unsqueeze(1),(sc_tar_mode_10 - sc_rel_mode_10).unsqueeze(1),
                            temp_p_h_2.unsqueeze(1),temp_p_h_5.unsqueeze(1), hi_pressure.unsqueeze(1),
                            temp_p_h_1_cab_heating.unsqueeze(1), lo_pressure.unsqueeze(1),aim_hi_pressure.unsqueeze(1)), dim=1)
        cab_heating_status_act_pos = self.cab_pos_model(dif_pressure,use_x2)

        # 结果输出限幅
        if not self.training:
            # compressor_speed = torch.round(compressor_speed).int()
            cab_heating_status_act_pos = torch.round(cab_heating_status_act_pos).int()
            compressor_speed = torch.clamp(compressor_speed, max=self.compressor_speed_max,
                                           min=self.compressor_speed_min)
            cab_heating_status_act_pos = torch.clamp(cab_heating_status_act_pos,
                                                     max=self.cab_heating_status_act_pos_max,
                                                     min=self.cab_heating_status_act_pos_min)

            com_speed_min_mask = compressor_speed < com_speed_min
            compressor_speed[com_speed_min_mask] = com_speed_min[com_speed_min_mask]
            compressor_speed[com_speed_zeros_mask] = 0
        return torch.cat((compressor_speed.unsqueeze(1),cab_heating_status_act_pos.unsqueeze(1)),dim=1)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.5):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        y = self.relu(out + res)
        # 检查中间结果是否包含 NaN
        if torch.isnan(y).any():
            print("NaN detected in TemporalBlock output.")

        return y



class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.5):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        y = self.network(x)
        # 在 TCN 输出后也检查是否包含 NaN
        assert not torch.isnan(y).any(), "NaN detected in TCN output."
        return y


if __name__ == '__main__':
    # temp_p_h_1_cab_heating = torch.randn(1024,1).numpy()
    # lo_pressure = torch.randn(1024, 1).numpy()
    # temp_p_h_2 = torch.randn(1024, 1).numpy()
    # hi_pressure = torch.randn(1024, 1).numpy()
    # # 压焓图1的焓值，排气温度焓值，制热
    # h_p_h_1_cab_heating = cal_h(temp_p_h_1_cab_heating, lo_pressure, states='gas')
    # # 压焓图2的焓值，进气温度焓值，制热
    # h_p_h_2_cab_heating = cal_h(temp_p_h_2, hi_pressure, states='gas')
    A = [375,225]
    B = [1777,1234]
    print(com_speed_min_bilinear_interpolation(A, B))

# 1