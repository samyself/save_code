import torch
import torch.nn as nn
import torch.nn.init as init

from torch.onnx.symbolic_opset11 import unsqueeze



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
    def __init__(self, input_size=6,output_size=1, hidden_size=None, dropout = 0.2):
        super(MyCabPosModel,self).__init__()
        # 开度建模
        if hidden_size is None:
            hidden_size = [1024, 1024]
        self.Cd_A0_row = nn.Parameter(torch.rand(1), requires_grad=True)
        self.cab_pos_k = nn.Parameter(torch.rand(1), requires_grad=True)
        self.cab_pos_n = nn.Parameter(torch.ones(1), requires_grad=True)

        # 转速偏差b2
        layers = []
        for i in range(len(hidden_size)):
            in_channels = input_size if i == 0 else hidden_size[i-1]
            out_channels = hidden_size[i]
            layers += [nn.Linear(in_channels, out_channels),nn.ReLU(),nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_size[-1], output_size)]
        self.linear1 = nn.Sequential(*layers)
        init_weights_xavier_uniform(self.linear1)

    def forward(self, use_x,dif_pressure):
        bais = self.linear1(use_x)
        m_val = (torch.clamp(self.Cd_A0_row * dif_pressure, min=0.0)) ** (1 / 2)
        cab_heating_status_act_pos = self.cab_pos_k * (m_val ** self.cab_pos_n) + bais[:,0]
        return cab_heating_status_act_pos

class MyComSpeedModel(nn.Module):
    def __init__(self, input_size=6, output_size=1, hidden_size=None, dropout = 0.5):
        super(MyComSpeedModel,self).__init__()
        # 压缩机建模
        # 不定系数n
        if hidden_size is None:
            hidden_size = [256, 256]
        self.com_speed_n = nn.Parameter(torch.tensor(2.0), requires_grad=True)
        # 功率系数k1
        self.com_speed_k1 = nn.Parameter(torch.rand(1), requires_grad=True)

        # 功率偏差b1
        layers = []
        for i in range(len(hidden_size)):
            in_channels = input_size if i == 0 else hidden_size[i-1]
            out_channels = hidden_size[i]
            layers += [nn.Linear(in_channels, out_channels),nn.ReLU(),nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_size[-1], output_size)]

        self.com_speed_b1 = nn.Sequential(*layers)

        # 转速系数k2
        self.com_speed_k2 = nn.Parameter(torch.rand(1), requires_grad=True)
        # 转速偏差b2
        layers = []
        for i in range(len(hidden_size)):
            in_channels = input_size if i == 0 else hidden_size[i-1]
            out_channels = hidden_size[i]
            layers += [nn.Linear(in_channels, out_channels),nn.ReLU(),nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_size[-1], output_size)]
        self.com_speed_b2 = nn.Sequential(*layers)

        init_weights_xavier_uniform(self.com_speed_b1)
        init_weights_xavier_uniform(self.com_speed_b2)

    def forward(self, dif_temp_p_h, h1_h2, use_x1):
        # 压缩机功率
        W_com = torch.clamp(self.com_speed_n / torch.clamp(self.com_speed_n - 1, min=1e-1) * (
                self.com_speed_k1 * dif_temp_p_h.unsqueeze(1)) + self.com_speed_b1(use_x1), min=0.0)

        compressor_speed = W_com[:, 0] / h1_h2[:,0] * self.com_speed_k2 + self.com_speed_b2(use_x1)[:, 0]
        return compressor_speed




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

        self.compressor_speed_model = MyComSpeedModel(input_size=6, output_size=1, hidden_size=[256,256], dropout = 0.5)
        self.cab_pos_model = MyCabPosModel(input_size=6,output_size=1, hidden_size=[1024,1024], dropout = 0.2)




    def forward(self, x):
        # 压缩机排气温度、内冷温度、饱和高压、压缩机进气温度、饱和低压、目标饱和高压、目标过冷度、目标过热度
        # 压缩机转速、膨胀阀开度

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
        # # 压焓图1的焓值，排气温度焓值，制热
        # h_p_h_1_cab_heating = cal_h(temp_p_h_1_cab_heating.numpy(), lo_pressure.numpy(), states='gas')
        # # 压焓图2的焓值，进气温度焓值，制热
        # h_p_h_2_cab_heating = cal_h(temp_p_h_2.numpy(), hi_pressure.numpy(), states='gas')
        # 焓值差
        # h1_h2 = torch.clamp((torch.from_numpy(h_p_h_1_cab_heating) - torch.from_numpy(h_p_h_2_cab_heating)), min=1e-6)
        h1_h2 = torch.ones(temp_p_h_2.shape[0], 1)
        # 计算最小理论压缩机转速
        com_speed_min = com_speed_min_bilinear_interpolation(lo_pressure, hi_pressure)

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

        compressor_speed = self.compressor_speed_model(dif_temp_p_h, h1_h2, use_x1)

        """
            膨胀阀开度前向传播
        """
        # 排气 压缩机温度差 内冷 饱高 压缩比 包和高压差
        use_x = torch.cat((temp_p_h_2.unsqueeze(1), temp_p_h_5.unsqueeze(1), hi_pressure.unsqueeze(1),
                           temp_p_h_1_cab_heating.unsqueeze(1), lo_pressure.unsqueeze(1), aim_hi_pressure.unsqueeze(1)), dim=1)
        cab_heating_status_act_pos = self.cab_pos_model(use_x,dif_pressure)

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
            compressor_speed[com_speed_min_mask] = com_speed_min[com_speed_min_mask].to(compressor_speed.dtype)
            compressor_speed[com_speed_zeros_mask] = 0
        return torch.cat((compressor_speed.unsqueeze(1),cab_heating_status_act_pos.unsqueeze(1)),dim=1)

# 1