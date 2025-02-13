import torch
import torch.nn as nn
import torch.nn.init as init
from model.data_utils_common import inter2D,inter1D

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


def norm(tensor,max,min):
    output = (tensor - min)/(max - min)
    return output

def renorm(tensor,max,min):
    output = tensor*(max - min) + min
    return output

def init_weights_xavier_uniform(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

# 模型输出确定
def get_acpid_discount(last_ac_pid_out, offset, hi_pressure, temp_p_h_2, raito_press, lo_pressure):

    if hi_pressure > 170 and hi_pressure < 2050 and temp_p_h_2 < 105 and raito_press < 13 and lo_pressure > 120:
        if offset > 800:
            return last_ac_pid_out + 800.0
        elif offset < -800:
            return last_ac_pid_out - 800.0
        else:
            return last_ac_pid_out + offset

    R1 = 1
    f1 = 1
    # 高压低保护 Ramp:170 OK:150 OFF:110
    if hi_pressure >= 110 and hi_pressure < 150:
        f1 = (hi_pressure - 150) / (110 - 150)
    elif hi_pressure >= 150 and hi_pressure < 170:
        R1 = (hi_pressure - 170) / (150 - 170)

    R2 = 1
    f2 = 1
    # 高压低保护 Ramp:2050 OK:2200 OFF:2500
    if hi_pressure > 2200 and hi_pressure <= 2500:
        f2 = (hi_pressure - 2200) / (2500 - 2200)
    elif hi_pressure > 2050 and hi_pressure <= 2200:
        R2 = (hi_pressure - 2050) / (2200 - 2050)

    R3 = 1
    f3 = 1
    # 排温高保护 Ramp:105 OK:120 OFF:125
    if temp_p_h_2 > 120 and temp_p_h_2 <= 125:
        f3 = (temp_p_h_2 - 120) / (125 - 120)
    elif temp_p_h_2 > 105 and temp_p_h_2 <= 120:
        R3 = (temp_p_h_2 - 105) / (120 - 105)

    R4 = 1
    f4 = 1
    # 压比保护 Ramp:13 OK:14 OFF:19
    if raito_press > 14 and raito_press <= 19:
        f4 = (raito_press - 14) / (19 - 14)
    elif raito_press > 13 and raito_press <= 14:
        R4 = (raito_press - 13) / (14 - 13)

    R5 = 1
    f5 = 1
    # 低压低保护 Ramp:120 OK:110 OFF:100
    if lo_pressure >= 100 and lo_pressure < 110:
        f5 = (lo_pressure - 110) / (100 - 110)
    elif lo_pressure >= 110 and lo_pressure < 120:
        R5 = (lo_pressure - 110) / (120 - 110)


    f = [f1, f2, f3, f4, f5]
    R = [R1, R2, R3, R4, R5]

    if min(f) == 1:
        R_min = min(R)
        R_list = [0, 0.40, 0.60, 0.80, 1.00]
        com_speed_list = [500, 100, 10, 5, 1]
        min_change = inter1D(R_list, com_speed_list, R_min)
        offset = torch.relu(offset - min_change) + min_change
        return last_ac_pid_out + offset
    else:
        f_min = min(f)
        return last_ac_pid_out * f_min



class MyModel(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.temp_amb_max = 55.0
        self.temp_amb_min = -40.0
        self.cab_set_temp_max = 32.0
        self.cab_set_temp_min = 18.0
        self.cab_req_temp_max = 60.0
        self.cab_req_temp_min = -30.0
        self.lo_pressure_max = 700.0
        self.lo_pressure_min = 100.0
        self.hi_pressure_max = 2400.0
        self.hi_pressure_min = 900.0

        self.last_ac_pid_out_hp = None

        # 高压差-Kp表
        self.PID_Kp_list = torch.tensor([0.5, 0.3417968, 0.3417968, 0.3417968, 0.3417968,0.5], device=device)
        self.diffpress_Kp_list = torch.tensor([-10, -2.5, -1, 1, 2.5, 10], device=device)
        # 高压差-Ki表
        self.PID_Ki_list = torch.tensor([0.1503906, 0.1503906, 0.1503906, 0.1503906, 0.1503906, 0.1503906], device=device)
        self.diffpress_Ki_list = torch.tensor([-10, -2.5, -1, 1, 2.5, 10], device=device)

        # 设定温度
        self.temp_set = torch.tensor([18.0, 20, 22, 24, 26, 28, 30, 31.5, 32], device=device)
        # 环境温度
        self.temp_envr = torch.tensor([-30.0, -20, -10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], device=device)
        # 第二个维度(环境温度)：-30.0,-20,-10,0,5,10,15,20,25,30,35,40,45,50    第一个维度(设定温度)
        self.CabinSP_table = torch.tensor([[17.0, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17],  # 18
                                           [20, 20, 19.5, 19.5, 19.5, 19, 19, 19, 18.5, 18.5, 18, 18, 18, 18],  # 20
                                           [22, 22, 22, 22.5, 22.5, 22.5, 22, 22, 21, 21, 21, 21, 20.5, 20],  # 22
                                           [24, 24.5, 25.5, 25.5, 26, 26, 25.5, 25, 24.5, 24, 23.5, 23, 23, 23],  # 24
                                           [27, 26.5, 27, 27.5, 28, 28, 27.5, 27, 26.5, 26, 25.5, 26, 26, 26],  # 26
                                           [29, 28.5, 28.5, 29.5, 30, 30, 29.5, 29, 29, 29, 28, 28, 29, 29],  # 28
                                           [31, 30.5, 30.5, 31.5, 32, 32, 32, 31, 31, 31, 31, 31, 31, 31],  # 30
                                           [32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 32, 32],  # 31.5
                                           [32, 32, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, ]], device=device)  # 32

        # 插值表
        self.lo_press_table = torch.tensor([100.0, 150, 200, 250, 300, 350, 400, 450, 500, 550], device=device)
        self.high_press_table = torch.tensor([200.0, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000], device=device)

        # 第二个维度(高压压力)：200,400,600,800,1000,1200,1400,1600,1800,2000                     第一个维度(低压压力)
        self.com_speed_min_table = torch.tensor([[2000.0, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000],  # 100
                                            [1600, 1600, 1600, 1600, 1600, 1700, 1800, 1900, 2000, 2000],  # 150
                                            [1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1600, 2000],  # 200
                                            [900, 900, 950, 1000, 1050, 1100, 1150, 1200, 1600, 2000],  # 250
                                            [800, 800, 800, 800, 900, 1000, 1100, 1200, 1600, 2000],  # 300
                                            [800, 800, 800, 800, 800, 900, 1050, 1200, 1600, 2000],  # 350
                                            [800, 800, 800, 800, 800, 800, 1000, 1200, 1600, 2000],  # 400
                                            [800, 800, 800, 800, 800, 800, 950, 1200, 1600, 2000],  # 450
                                            [800, 800, 800, 800, 800, 800, 900, 1200, 1600, 2000],  # 500
                                            [800, 800, 800, 800, 800, 800, 850, 1200, 1600, 2000]], device=device)  # 550


        self.AC_Kp_Rate =  nn.Parameter(torch.rand(1, device=device), requires_grad=True)
        self.Kd_paras = torch.tensor([0.048828], device=device)

        self.Ivalue = torch.tensor([0.0], device=device)
        self.last_Pout = torch.tensor([0.0], device=device)
        self.last_Dout = torch.tensor([0.0], device=device)

        self.last_diff_hp = 0


        self.MLP1 = MLP(input_size=4, hidden_sizes=[512, 512], output_size=1,use_batchnorm=False,use_dropout=False)
        # self.MLP2 = MLP(input_size=11, hidden_sizes=[256, 256], output_size=1,use_batchnorm=False,use_dropout=True)

        init_weights_xavier_uniform(self.MLP1)
        # init_weights_xavier_uniform(self.MLP2)

        # self.Iout_bais = MLP(input_size=4, hidden_sizes=[512, 512], output_size=1,use_batchnorm=False,use_dropout=False)
        # self.Diffout_bais = MLP(input_size=4, hidden_sizes=[512, 512], output_size=1, use_batchnorm=False,
        #                      use_dropout=False)
        # init_weights_xavier_uniform(self.Iout_bais)
        # init_weights_xavier_uniform(self.Diffout_bais)

        self.parameters_bias = MLP(input_size=14, hidden_sizes=[128, 64], output_size=1, use_batchnorm=False)
        self.parameters_k = MLP(input_size=1, hidden_sizes=[8, 8], output_size=1, use_batchnorm=False)

        init_weights_xavier_uniform(self.parameters_bias)
        init_weights_xavier_uniform(self.parameters_k)
        self.K_high_pressure = 0.0244

        self.offset_change_max = 800.0
        self.offset_change_min = -800.0
        self.last_offset = 0

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # last_ac_pid_out_hp = x[:, 0]
        last_ac_pid_out_hp = self.last_ac_pid_out_hp

        # 当前环境温度
        temp_amb = x[:, 1]

        # 当前主驾驶设定温度
        cab_fl_set_temp = x[:, 2]
        # 当前副驾驶设定温度
        cab_fr_set_temp = x[:, 3]


        # 上一时刻饱和低压
        lo_pressure = x[:, 4]
        # 上一时刻饱和高压
        hi_pressure = x[:, 5]
        # 目标饱和低压
        aim_lo_pressure = x[:, 6]
        # 目标饱和高压
        aim_hi_pressure = x[:, 7]

        # 当前乘务舱温度
        temp_incar = x[:, 8]
        # 电池请求冷却液温度
        temp_battery_req = x[:, 9]
        # 冷却液进入电池的温度
        temp_coolant_battery_in = x[:, 10]

        # ac_kp_rate_last
        ac_kp_rate_last = x[:,11]

        # 压缩机排气温度
        temp_p_h_2 = x[:,12]
        # # 内冷温度
        # temp_p_h_5 = x[:,13]
        # # 压缩机进气温度
        # temp_p_h_1_cab_heating = x[:,14]


        # 压比
        raito_press = hi_pressure/lo_pressure

       #高压偏差
        diff_hi_pressure = (aim_hi_pressure - hi_pressure) * self.K_high_pressure
        diff_hi_pressure_real = (aim_hi_pressure - hi_pressure) * self.parameters_k(diff_hi_pressure)

        # 低压偏差
        AC_KpRate = ac_kp_rate_last.detach()


        Kp = inter1D(self.diffpress_Kp_list, self.PID_Kp_list, diff_hi_pressure)*AC_KpRate
        Ki = inter1D(self.diffpress_Ki_list, self.PID_Ki_list, diff_hi_pressure)*AC_KpRate
        Kd = self.Kd_paras*AC_KpRate

        com_speed_min = inter2D(self.lo_press_table, self.high_press_table, self.com_speed_min_table, lo_pressure, hi_pressure)


        last_Ivalue = self.Ivalue.detach()
        if abs(diff_hi_pressure) < 0.39063:
            self.Ivalue = torch.zeros_like(self.Ivalue) * diff_hi_pressure_real
        else:
            # self.Ivalue = (self.Ivalue + diff_hi_pressure)/2
            self.Ivalue = (self.Ivalue + diff_hi_pressure_real) / 2

        # if len(last_ac_pid_out_hp.shape) == 1:
        #     last_ac_pid_out_hp = last_ac_pid_out_hp.unsqueeze(1)
        last_ac_pid_out_hp = last_ac_pid_out_hp.view(-1, 1)
        Iout = Ki * (self.Ivalue + last_Ivalue)
        # Pout = Kp * diff_hi_pressure
        Pout = Kp * diff_hi_pressure_real

        Dout = 0.0
        Diffout = (Pout + Dout) - (self.last_Pout + self.last_Dout)
        offset = Iout + Diffout
        # offset = offset * self.parameters_k(x[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14]]) + self.parameters_bias(x[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14]])
        # offset = torch.clamp(offset, min=self.offset_change_min, max=self.offset_change_max)

        # ac_pid_out_hp = get_acpid_discount(last_ac_pid_out_hp, offset, hi_pressure, temp_p_h_2, raito_press, lo_pressure)
        ac_pid_out_hp = last_ac_pid_out_hp + offset
        # ac_pid_out_hp = torch.clamp(ac_pid_out_hp, min=com_speed_min.item(), max=8000)

        # ac_pid_out_hp = self.k*ac_pid_out_hp + self.bais

        self.last_diff_hp = diff_hi_pressure
        self.last_Pout = Pout
        self.last_Dout = Dout
        self.last_offset = ac_pid_out_hp.detach() - last_ac_pid_out_hp
        self.last_ac_pid_out_hp = ac_pid_out_hp.detach()

        return ac_pid_out_hp