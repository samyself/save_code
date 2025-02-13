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
def get_acpid_discount(offset, hi_pressure, temp_p_h_2, raito_press, lo_pressure):
    # 高压低保护 Ramp:170 OK:150 OFF:110
    HP_lo_OFF = torch.tensor(110.0)
    HP_lo_OK = torch.tensor(150.0)
    HP_lo_Ramp = torch.tensor(170.0)
    # 高压高保护 Ramp:2050 OK:2200 OFF:2500
    HP_hi_OFF = torch.tensor(2500.0)
    HP_hi_OK = torch.tensor(2200.0)
    HP_hi_Ramp = torch.tensor(2050.0)
    # 排气温度高保护 Ramp:95 OK:105 OFF:110
    TP2_hi_OFF = torch.tensor(110.0)
    TP2_hi_OK = torch.tensor(105.0)
    TP2_hi_Ramp = torch.tensor(95.0)
    #压比高保护 Ramp:13 OK:14 OFF:16
    RP_hi_OFF = torch.tensor(16.0)
    RP_hi_OK = torch.tensor(14.0)
    RP_hi_Ramp = torch.tensor(13.0)
    # 低压低保护 Ramp:120 OK:105 OFF:95
    LP_lo_OFF = torch.tensor(95.0)
    LP_lo_OK = torch.tensor(105.0)
    LP_lo_Ramp = torch.tensor(120.0)

    hi_pressure = torch.clamp(hi_pressure, min=HP_lo_OFF, max=HP_hi_OFF)
    temp_p_h_2 = torch.clamp(temp_p_h_2, max=TP2_hi_OFF)
    raito_press = torch.clamp(raito_press, max=RP_hi_OFF)
    lo_pressure = torch.clamp(lo_pressure, min=LP_lo_OFF)
    R = -1
    f = -1
    # 没有触发保护
    if hi_pressure > HP_lo_Ramp and hi_pressure < HP_hi_Ramp and temp_p_h_2 < TP2_hi_Ramp and raito_press < RP_hi_Ramp and lo_pressure > LP_lo_Ramp:
        return f, R

    R1 = -R
    f1 = -f
    # 高压低保护 Ramp:170 OK:150 OFF:110
    if hi_pressure >= HP_lo_OFF and hi_pressure < HP_lo_OK:
        f1 = (hi_pressure - HP_lo_OK) / (HP_lo_OFF - HP_lo_OK)
    elif hi_pressure >= HP_lo_OK and hi_pressure < HP_lo_Ramp:
        R1 = (hi_pressure - HP_lo_Ramp) / (HP_lo_OK - HP_lo_Ramp)

    R2 = -R
    f2 = -f
    # 高压低保护 Ramp:2050 OK:2200 OFF:2500
    if hi_pressure > HP_hi_OK and hi_pressure <= HP_hi_OFF:
        f2 = (hi_pressure - HP_hi_OK) / (HP_hi_OFF - HP_hi_OK)
    elif hi_pressure > HP_hi_Ramp and hi_pressure <= HP_hi_OK:
        R2 = (hi_pressure - HP_hi_Ramp) / (HP_hi_OK - HP_hi_Ramp)

    R3 = -R
    f3 = -f
    # 排温高保护 Ramp:95 OK:105 OFF:110
    if temp_p_h_2 > TP2_hi_OK and temp_p_h_2 <= TP2_hi_OFF:
        f3 = (temp_p_h_2 - TP2_hi_OK) / (TP2_hi_OFF - TP2_hi_OK)
    elif temp_p_h_2 > TP2_hi_Ramp and temp_p_h_2 <= TP2_hi_OK:
        R3 = (temp_p_h_2 - TP2_hi_Ramp) / (TP2_hi_OK - TP2_hi_Ramp)

    R4 = -R
    f4 = -f
    # 压比保护 Ramp:13 OK:14 OFF:16
    if raito_press > RP_hi_OK and raito_press <= RP_hi_OFF:
        f4 = (raito_press - RP_hi_OK) / (RP_hi_OFF - RP_hi_OK)
    elif raito_press > RP_hi_Ramp and raito_press <= RP_hi_OK:
        R4 = (raito_press - RP_hi_Ramp) / (RP_hi_OK - RP_hi_Ramp)

    R5 = -R
    f5 = -f
    # 低压低保护 Ramp:120 OK:105 OFF:95
    if lo_pressure >= LP_lo_OFF and lo_pressure < LP_lo_OK:
        f5 = (lo_pressure - LP_lo_OK) / (LP_lo_OFF - LP_lo_OK)
    elif lo_pressure >= LP_lo_OK and lo_pressure < LP_lo_Ramp:
        R5 = (lo_pressure - LP_lo_Ramp) / (LP_lo_OK - LP_lo_Ramp)


    f = [f1, f2, f3, f4, f5]
    R = [R1, R2, R3, R4, R5]

    return min(f), min(R)


    # f_min = min(f)
    # # 约束变化幅度上限
    # if f_min < 1:
    #     R_min = min(R)
    #     R_list = [0, 0.40, 0.60, 0.80, 1.00]
    #     com_speed_list = [500, 100, 10, 5, 1]
    #     max_change = inter1D(R_list, com_speed_list, R_min)
    #     if abs(offset) > max_change:
    #         if offset >= 0:
    #             offset = max_change
    #         else:
    #             offset = -max_change
    #     return offset, f_min, 800.0
    #
    # else:
    #     f_min = min(f)
    #     return offset,f_min,0.0



class MyModel(nn.Module):
    def __init__(self, device='cpu', car_type='modena'):
        super().__init__()
        self.car_type = car_type

        # Cabin第一个参数 [:,0]是环境温度temp_amb， CabinP1_table_modena[:,1]是CabinP1
        self.CabinP1_table_modena = torch.tensor(
            [[-20.0, 3.0], [-10.0, 2.0], [0.0, 0.88], [10.0, 0.38], [25.0, 0.7], [30.0, 1.0], [35.0, 1.31], [40.0, 2.5],
             [45.0, 3.0]])
        self.CabinP1_table_lemans = torch.tensor(
            [[-20.0, 3.0], [-10.0, 2.0], [0.0, 0.875], [10.0, 0.375], [25.0, 0.69531], [30.0, 1.0], [35.0, 1.3125],
             [40.0, 2.5], [45.0, 3.0]])

        # 设定温度
        self.temp_set = torch.tensor([18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 31.5, 32.0])
        # 环境温度
        self.temp_envr = torch.tensor([-30.0, -20.0, -10.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])
        # 第二个维度(环境温度)：-30.0,-20,-10,0,5,10,15,20,25,30,35,40,45,50    第一个维度(设定温度)
        self.CabinSP_table = torch.tensor([[17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0],  # 18
                                           [20.0, 20.0, 19.5, 19.5, 19.5, 19.0, 19.0, 19.0, 18.5, 18.5, 18.0, 18.0, 18.0, 18.0],  # 20
                                           [22.0, 22.0, 22.0, 22.5, 22.5, 22.5, 22.0, 22.0, 21.0, 21.0, 21.0, 21.0, 20.5, 20.0],  # 22
                                           [24.0, 24.5, 25.5, 25.5, 26.0, 26.0, 25.5, 25.0, 24.5, 24.0, 23.5, 23.0, 23.0, 23.0],  # 24
                                           [27.0, 26.5, 27.0, 27.5, 28.0, 28.0, 27.5, 27.0, 26.5, 26.0, 25.5, 26.0, 26.0, 26.0],  # 26
                                           [29.0, 28.5, 28.5, 29.5, 30.0, 30.0, 29.5, 29.0, 29.0, 29.0, 28.0, 28.0, 29.0, 29.0],  # 28
                                           [31.0, 30.5, 30.5, 31.5, 32.0, 32.0, 32.0, 31.0, 31.0, 31.0, 31.0, 31.0, 31.0, 31.0],  # 30
                                           [32.0, 32.0, 32.0, 33.0, 33.0, 33.0, 33.0, 33.0, 33.0, 33.0, 33.0, 33.0, 32.0, 32.0],  # 31.5
                                           [32.0, 32.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0]])  # 32
        # Cabin第二个参数，CabinP2_table_modena[:,0]DVT_CabinErr(取DVT_CabinErr_FL/FR的较小值)，CabinP2_table_modena[:,1]是CabinP2
        self.CabinP2_table = torch.tensor(
            [[-20.0, 3.0], [-15.0, 2.0], [-10.0, 1.0], [-5.0, 0.50], [0.0, 0.40], [5.0, 0.50], [10.0, 1.5],
             [15.0, 3.0], [20.0, 6.0]])
        # Chiller参数，ChillerP1_table_modena[:,0]为DCT_ChillerTempErr= 电池请求温度BattWTReqOpt - 电池进口水温HvBattInletCooltT，CabinP2_table_modena[:,1]是ChillerP1
        self.ChillerP1_table_modena = torch.tensor(
            [[-10.0, 4.0], [-5.0, 2.0], [-1.0, 1.0], [0.0, 0.40], [1.0, 0.40], [2.0, 0.40], [5.0, 1.0], [10.0, 2.0],
             [20.0, 3.0], [30.0, 4.0]])
        self.ChillerP1_table_lemans = torch.tensor(
            [[-10.0, 6.0], [-5.0, 3.0], [-1.0, 1.5], [0.0, 0.79980], [1.0, 0.79980], [2.0, 1.20019], [5.0, 1.5],
             [10.0, 3.0],
             [20.0, 4.5], [30.0, 6.0]])

        # 蒸发器温度决定最小调整速率
        self.EvapTemp_table = torch.tensor([[2.0, 1.0], [3.0, 10.0], [5.0, 50.0], [6.0, 500.0]])

        self.MLP1 = MLP(input_size=2, hidden_sizes=[64, 64], output_size=1, use_batchnorm=False, use_dropout=False)
        # self.MLP2 = MLP(input_size=4, hidden_sizes=[64, 64], output_size=1, use_batchnorm=False, use_dropout=False)
        self.bais = torch.tensor(0.39306798577308655)

        init_weights_xavier_uniform(self.MLP1)


        # 高压差-Kp表
        self.PID_Kp_list_HP = torch.tensor([0.5, 0.3417968, 0.3417968, 0.3417968, 0.3417968,0.5], device=device)
        self.diffpress_Kp_list_HP = torch.tensor([-10.0, -2.5, -1.0, 1.0, 2.5, 10.0], device=device)
        # 高压差-Ki表
        self.PID_Ki_list_HP = torch.tensor([0.1503906, 0.1503906, 0.1503906, 0.1503906, 0.1503906, 0.1503906], device=device)
        self.diffpress_Ki_list_HP = torch.tensor([-10.0, -2.5, -1.0, 1.0, 2.5, 10.0], device=device)


        # 低压差-Kp表
        self.PID_Kp_list_LP = torch.tensor([0.49, 0.49, 0.05, 0.05, 0.49,0.49], device=device)
        self.diffpress_Kp_list_LP = torch.tensor([-10.0, -2.5, -1.0, 1.0, 2.5, 10.0], device=device)
        # 高压差-Ki表
        self.PID_Ki_list_LP = torch.tensor([0.15, 0.13, 0.12, 0.06, 0.06, 0.09], device=device)
        self.diffpress_Ki_list_LP = torch.tensor([-10.0, -2.5, -1.0, 1.0, 2.5, 10.0], device=device)

        # 插值表
        self.lo_press_table = torch.tensor([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0], device=device)
        self.high_press_table = torch.tensor([200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0], device=device)

        # 第二个维度(高压压力)：200,400,600,800,1000,1200,1400,1600,1800,2000                     第一个维度(低压压力)
        self.com_speed_min_table_modena = torch.tensor([[2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0],  # 100
                                            [1600.0, 1600.0, 1600.0, 1600.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0, 2000.0],  # 150
                                            [1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1600.0, 2000.0],  # 200
                                            [900.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1600.0, 2000.0],  # 250
                                            [800.0, 800.0, 800.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1600.0, 2000.0],  # 300
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 900.0, 1050.0, 1200.0, 1600.0, 2000.0],  # 350
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 1000.0, 1200.0, 1600.0, 2000.0],  # 400
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 950.0, 1200.0, 1600.0, 2000.0],  # 450
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 900.0, 1200.0, 1600.0, 2000.0],  # 500
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 850.0, 1200.0, 1600.0, 2000.0]], device=device)  # 550

        # 第二个维度(高压压力)：200,400,600,800,1000,1200,1400,1600,1800,2000                     第一个维度(低压压力)
        self.com_speed_min_table_lemans_5 = torch.tensor([[1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1700.0, 2400.0, 3000.0, 3000.0, 3000.0],  # 100
                                            [900.0, 900.0, 900.0, 900.0, 900.0, 950.0, 1500.0, 2000.0, 2500.0, 3000.0],  # 150
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 850.0, 900.0, 950.0, 1000.0, 1500.0],  # 200
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 850.0, 900.0, 1000.0, 1400.0],  # 250
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 900.0, 1000.0, 1300.0],  # 300
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 900.0, 1000.0, 1300.0],  # 350
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 900.0, 1000.0, 1300.0],  # 400
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 900.0, 1000.0, 1350.0],  # 450
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 900.0, 1000.0, 1400.0],  # 500
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 900.0, 1000.0, 1450.0]], device=device)  # 550

        # 第二个维度(高压压力)：200,400,600,800,1000,1200,1400,1600,1800,2000                     第一个维度(低压压力)
        self.com_speed_min_table_lemans_1_2 = torch.tensor([[2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0],  # 100
                                            [1600.0, 1600.0, 1600.0, 1600.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0, 2000.0],  # 150
                                            [1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1600.0, 2000.0],  # 200
                                            [900.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1600.0, 2000.0],  # 250
                                            [800.0, 800.0, 800.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1600.0, 2000.0],  # 300
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 900.0, 1050.0, 1200.0, 1600.0, 2000.0],  # 350
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 1000.0, 1200.0, 1600.0, 2000.0],  # 400
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 950.0, 1200.0, 1600.0, 2000.0],  # 450
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 900.0, 1200.0, 1600.0, 2000.0],  # 500
                                            [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 850.0, 1200.0, 1600.0, 2000.0]], device=device)  # 550


        # 环境温度确定下限
        self.com_speed_min_temp_amb_lemans_5 = torch.tensor([[10.0,800.0], [15.0,1000.0], [20.0, 1200.0]], device=device)
        self.com_speed_min_temp_amb_lemans_1_2 = torch.tensor([[10.0, 800.0], [15.0, 1200.0], [20.0, 1500.0]], device=device)


        self.Kd_paras_HP = torch.tensor([0.048828], device=device)
        self.Kd_paras_LP = torch.tensor([0.0097656 ], device=device)

        self.Ivalue = torch.tensor([0.0], device=device)

        # 蒸发器温度

        self.last_Pout = torch.tensor([0.0], device=device)
        self.last_Dout = torch.tensor([0.0], device=device)

        self.last_diff_hp = 0

        self.K_high_pressure = torch.tensor([0.0244], device=device)
        self.K_low_pressure = torch.tensor([0.0996], device=device)

        self.offset_change_max = torch.tensor([800.0], device=device)
        self.offset_change_min = torch.tensor([-800.0], device=device)
        self.last_offset = torch.tensor([0.0], device=device)

        self.num_LP = torch.tensor([0.0], device=device)
        self.last_AC_Percent_Req = None
        self.last_AC_RPM_Req = None

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        last_ac_pid_out_hp = x[:, 0]
        # last_ac_pid_out_hp = self.last_ac_pid_out_hp

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

        # 蒸发器温度
        temp_evap = x[:,15]
        # 空调模式
        hp_mode_ac = x[:,16]
        # 制冷模式下膨胀阀开度
        cab_cooling_status_act_pos = x[:,17]


        # ac_pid_pout_hp = x[:,16]
        # # Ac_pid_iout
        # ac_pid_iout_hp = x[:,17]
        # # Ac_pid_dout
        # ac_pid_dout_hp = x[:,18]

        # if abs(ac_pid_pout_hp) >= 1:
        #     ac_pid_pout_hp1 = ac_pid_pout_hp


        # 压比
        raito_press = hi_pressure/lo_pressure

       #高压偏差
        diff_hi_pressure = (aim_hi_pressure - hi_pressure) * self.K_high_pressure

        # 低压偏差
        diff_lo_pressure = (lo_pressure - aim_lo_pressure) * self.K_low_pressure


        # (1) 求AC_KpRate
        if 'modena' in self.car_type:
            CabinP1_table = self.CabinP1_table_modena
        elif 'lemans' in self.car_type:
            CabinP1_table = self.CabinP1_table_lemans

        AC_KpRateCabin1 = inter1D(CabinP1_table[:, 0], CabinP1_table[:, 1], temp_amb)

        #   DVT_CabinErr_FL = f(主驾设定温度, 环境温度) 查表 - SEN_Incar(乘员舱温度)
        DVT_CabinErr_FL = inter2D(self.temp_set, self.temp_envr, self.CabinSP_table, cab_fl_set_temp,
                                  temp_amb) - temp_incar.view(-1, 1)
        #   DVT_CabinErr_FR = f(副驾设定温度, 环境温度) 查表 - SEN_Incar(乘员舱温度)
        DVT_CabinErr_FR = inter2D(self.temp_set, self.temp_envr, self.CabinSP_table, cab_fr_set_temp,
                                  temp_amb) - temp_incar.view(-1, 1)

        DVT_CabinErr = torch.min(DVT_CabinErr_FL, DVT_CabinErr_FR)

        AC_KpRateCabin2 = inter1D(self.CabinP2_table[:, 0], self.CabinP2_table[:, 1], DVT_CabinErr)

        AC_KpRateCabin = torch.min(AC_KpRateCabin1, AC_KpRateCabin2)

        # DCT_DCT_ChillerTempErr = HP_Mode_BattWTReqOpt（电池请求冷却的温度） - SEN_CoolT_Battln（冷却液进入电池的温度）
        DCT_ChillerTempErr = temp_battery_req - temp_coolant_battery_in

        if 'modena' in self.car_type:
            ChillerP1_table = self.ChillerP1_table_modena
        else:
            ChillerP1_table = self.ChillerP1_table_lemans

        AC_KpRateChiller = inter1D(ChillerP1_table[:, 0], ChillerP1_table[:, 1], DCT_ChillerTempErr)

        use_x1 = torch.cat((AC_KpRateCabin.view(-1, 1), AC_KpRateChiller.view(-1, 1)), dim=1)

        AC_KpRate = self.MLP1(use_x1) + self.bais
        # AC_KpRate = ac_kp_rate_last
        try:
            Kp_HP = inter1D(self.diffpress_Kp_list_HP, self.PID_Kp_list_HP, diff_hi_pressure)*AC_KpRate
            Ki_HP = inter1D(self.diffpress_Ki_list_HP, self.PID_Ki_list_HP, diff_hi_pressure)*AC_KpRate
            Kd_HP = self.Kd_paras_HP*AC_KpRate

            Kp_LP = inter1D(self.diffpress_Kp_list_LP, self.PID_Kp_list_LP, diff_lo_pressure)*AC_KpRate
            Ki_LP = inter1D(self.diffpress_Ki_list_LP, self.PID_Ki_list_LP, diff_lo_pressure)*AC_KpRate
            Kd_LP = self.Kd_paras_LP*AC_KpRate
        except:
            print(diff_hi_pressure.shape)

        if 'modena' in self.car_type:
            com_speed_min_table = self.com_speed_min_table_modena
            com_speed_min_temp_amb = self.com_speed_min_temp_amb_lemans_5
        elif 'lemans_1' in self.car_type or 'lemans_2' in self.car_type:
            com_speed_min_table = self.com_speed_min_table_lemans_1_2
            com_speed_min_temp_amb = self.com_speed_min_temp_amb_lemans_1_2
        elif 'lemans_5' in self.car_type:
            com_speed_min_table = self.com_speed_min_table_lemans_5
            com_speed_min_temp_amb = self.com_speed_min_temp_amb_lemans_5
        else:
            com_speed_min_table = self.com_speed_min_table_lemans_5
            com_speed_min_temp_amb = self.com_speed_min_temp_amb_lemans_5

        com_speed_min1 = inter2D(self.lo_press_table, self.high_press_table, com_speed_min_table, lo_pressure, hi_pressure)
        # if temp_amb < 10 or temp_amb > 20:
        #     com_speed_min2 = com_speed_min1
        # else:
        temp_amb_limit = torch.clamp(temp_amb,min=10,max=20)
        com_speed_min2 = inter1D(com_speed_min_temp_amb[:,0], com_speed_min_temp_amb[:,1],temp_amb_limit)
        com_speed_min = torch.max(com_speed_min1, com_speed_min2)
        # com_speed_min = com_speed_min1
        # if self.Ivalue == None:
        #     self.Ivalue = Ki * diff_hi_pressure
        last_Ivalue = self.Ivalue

        # 高压PID输出值
        if abs(diff_hi_pressure) < 0.39063:
            Ivalue_HP = torch.zeros_like(self.Ivalue) * diff_hi_pressure
        else:
            Ivalue_HP = (self.Ivalue + diff_hi_pressure)/2

        last_ac_pid_out_hp = last_ac_pid_out_hp.view(-1, 1)
        Iout_HP = Ki_HP * (Ivalue_HP + last_Ivalue)/2
        Pout_HP = Kp_HP * diff_hi_pressure

        Dout = 0.0
        Diffout_HP = (Pout_HP + Dout) - (self.last_Pout + self.last_Dout)
        HP_AD = (Iout_HP + Diffout_HP)

        # 低压PID输出值
        if abs(diff_lo_pressure) < 0.59375:
            Ivalue_LP = torch.zeros_like(self.Ivalue) * diff_lo_pressure
        else:
            Ivalue_LP = (self.Ivalue + diff_lo_pressure)/2

        Iout_LP = Ki_LP * (Ivalue_LP + last_Ivalue)/2
        Pout_LP = Kp_LP * diff_lo_pressure

        Dout = 0.0
        Diffout_LP = (Pout_LP + Dout) - (self.last_Pout + self.last_Dout)
        LP_AD = (Iout_LP + Diffout_LP)

        if HP_AD <= LP_AD:
            mode = 'HP'
        else:
            mode = 'LP'
            self.num_LP = self.num_LP + 1

        if mode == 'HP':
            offset = HP_AD * 85.0
            Pout = Pout_HP
            self.Ivalue = Ivalue_HP
        else:
            offset = LP_AD * 85.0
            Pout = Pout_LP
            self.Ivalue = Ivalue_LP

            # 保护逻辑
        # f_min, R_min = get_acpid_discount(offset, hi_pressure, temp_p_h_2, raito_press, lo_pressure)
        # if R_min != -1.0 and R_min < 1.0:
        #     R_list = [0, 0.40, 0.60, 0.80, 1.00]
        #     com_speed_list = [500, 100, 10, 5, 1]
        #     max_change = inter1D(R_list, com_speed_list, R_min)
        #     max_change = torch.clamp(max_change, min=1.0)
        #     offset = torch.clamp(offset, min=-max_change, max=max_change)
        #     f = 1.0
        # elif f_min != -1.0 and f_min < 1.0:
        #     f = f_min
        #     offset = torch.clamp(offset, max=0.0)
        #     max_change = 500.0
        # else:
        #     f = 1.0
        #     offset = torch.clamp(offset, max=800.0)
        #     max_change = 500.0
        max_change = 500.0

        # 变化逻辑
        if last_ac_pid_out_hp + 100 > com_speed_min.item():
            if cab_cooling_status_act_pos > 0 and hp_mode_ac > 0:
                # 蒸发器温度查表
                temp_evap_up_max = inter1D(self.EvapTemp_table[:, 0], self.EvapTemp_table[:, 1], temp_evap)
                temp_evap_up_max = torch.clamp(temp_evap_up_max, min=1.0)
            else:
                temp_evap_up_max = torch.tensor(200.0)
            # 最大上升速度
            max_up_change_spd = min([torch.tensor(200), AC_KpRate * 78, max_change,temp_evap_up_max])
            # max_up_change_spd = min([torch.tensor(200), diff_ac_pid_out, AC_KpRate*78, temp_evap_up_max])
            # if max_up_change_spd < 0:
            #     print('max_up_change_spd', max_up_change_spd)
            # 最大下降速度
            max_down_change_spd = torch.tensor(-200.0)
        else:
            max_up_change_spd = torch.tensor(8500.0)
            max_down_change_spd = torch.tensor(-8500.0)

        # ac_pid_out_hp = (last_ac_pid_out_hp + offset) * f
        # ac_pid_out_hp = torch.clamp(ac_pid_out_hp, min=(last_ac_pid_out_hp + max_down_change_spd).item(), max=(last_ac_pid_out_hp + max_up_change_spd).item())

        offset = torch.clamp(offset, min=max_down_change_spd, max=max_up_change_spd.item())

        ac_pid_out_hp = (last_ac_pid_out_hp + offset)

        # 上下限
        # max_ac_pid_out_hp = torch.max(last_ac_pid_out_hp + max_change_up,torch.tensor(1700.0))
        # max_ac_pid_out_hp = torch.min(max_ac_pid_out_hp,torch.tensor(8000.0))

        if ac_pid_out_hp < com_speed_min.item() - 120:
            com_speed_min = torch.tensor([800.0])

        # ac_pid_out_hp = last_ac_pid_out_hp + offset
        ac_pid_out_hp = torch.clamp(ac_pid_out_hp, min=com_speed_min.item(), max=8000.0)

        # ac_pid_out_hp = self.k*ac_pid_out_hp + self.bais

        # self.last_diff_hp = diff_hi_pressure
        self.last_Pout = Pout
        self.last_Dout = Dout
        self.last_offset = ac_pid_out_hp.detach() - last_ac_pid_out_hp
        self.last_ac_pid_out_hp = ac_pid_out_hp.detach()

        if self.last_offset >= 800:
            self.last_offset = 0.0
            self.last_ac_pid_out_hp = last_ac_pid_out_hp.detach()
            ac_pid_out_hp = last_ac_pid_out_hp


        return torch.cat((ac_pid_out_hp.view(-1,1),Kp_HP.view(-1,1),Ki_HP.view(-1,1),Kd_HP.view(-1,1),Iout_HP.view(-1,1),Diffout_HP.view(-1,1),HP_AD.view(-1,1),
                          Kp_LP.view(-1,1),Ki_LP.view(-1,1),Kd_LP.view(-1,1),Iout_LP.view(-1,1),Diffout_LP.view(-1,1),LP_AD.view(-1,1)), dim=1)

if __name__ == '__main__':
    offset,f = get_acpid_discount(900, 1000, 105, 1, 1000)
    print(offset,f)