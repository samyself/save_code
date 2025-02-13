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

    R1 = R
    f1 = f
    # 高压低保护 Ramp:170 OK:150 OFF:110
    if hi_pressure >= HP_lo_OFF and hi_pressure < HP_lo_OK:
        f1 = (hi_pressure - HP_lo_OK) / (HP_lo_OFF - HP_lo_OK)
    elif hi_pressure >= HP_lo_OK and hi_pressure < HP_lo_Ramp:
        R1 = (hi_pressure - HP_lo_Ramp) / (HP_lo_OK - HP_lo_Ramp)

    R2 = R
    f2 = f
    # 高压低保护 Ramp:2050 OK:2200 OFF:2500
    if hi_pressure > HP_hi_OK and hi_pressure <= HP_hi_OFF:
        f2 = (hi_pressure - HP_hi_OK) / (HP_hi_OFF - HP_hi_OK)
    elif hi_pressure > HP_hi_Ramp and hi_pressure <= HP_hi_OK:
        R2 = (hi_pressure - HP_hi_Ramp) / (HP_hi_OK - HP_hi_Ramp)

    R3 = R
    f3 = f
    # 排温高保护 Ramp:95 OK:105 OFF:110
    if temp_p_h_2 > TP2_hi_OK and temp_p_h_2 <= TP2_hi_OFF:
        f3 = (temp_p_h_2 - TP2_hi_OK) / (TP2_hi_OFF - TP2_hi_OK)
    elif temp_p_h_2 > TP2_hi_Ramp and temp_p_h_2 <= TP2_hi_OK:
        R3 = (temp_p_h_2 - TP2_hi_Ramp) / (TP2_hi_OK - TP2_hi_Ramp)

    R4 = R
    f4 = f
    # 压比保护 Ramp:13 OK:14 OFF:16
    if raito_press > RP_hi_OK and raito_press <= RP_hi_OFF:
        f4 = (raito_press - RP_hi_OK) / (RP_hi_OFF - RP_hi_OK)
    elif raito_press > RP_hi_Ramp and raito_press <= RP_hi_OK:
        R4 = (raito_press - RP_hi_Ramp) / (RP_hi_OK - RP_hi_Ramp)

    R5 = R
    f5 = f
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
        self.temp_set = torch.tensor([18.0, 20, 22, 24, 26, 28, 30, 31.5, 32])
        # 环境温度
        self.temp_envr = torch.tensor([-30.0, -20, -10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        # 第二个维度(环境温度)：-30.0,-20,-10,0,5,10,15,20,25,30,35,40,45,50    第一个维度(设定温度)
        self.CabinSP_table = torch.tensor([[17.0, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17],  # 18
                                           [20, 20, 19.5, 19.5, 19.5, 19, 19, 19, 18.5, 18.5, 18, 18, 18, 18],  # 20
                                           [22, 22, 22, 22.5, 22.5, 22.5, 22, 22, 21, 21, 21, 21, 20.5, 20],  # 22
                                           [24, 24.5, 25.5, 25.5, 26, 26, 25.5, 25, 24.5, 24, 23.5, 23, 23, 23],  # 24
                                           [27, 26.5, 27, 27.5, 28, 28, 27.5, 27, 26.5, 26, 25.5, 26, 26, 26],  # 26
                                           [29, 28.5, 28.5, 29.5, 30, 30, 29.5, 29, 29, 29, 28, 28, 29, 29],  # 28
                                           [31, 30.5, 30.5, 31.5, 32, 32, 32, 31, 31, 31, 31, 31, 31, 31],  # 30
                                           [32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 32, 32],  # 31.5
                                           [32, 32, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36]])  # 32
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
        self.PID_Kp_list = torch.tensor([0.5, 0.3417968, 0.3417968, 0.3417968, 0.3417968,0.5], device=device)
        self.diffpress_Kp_list = torch.tensor([-10, -2.5, -1, 1, 2.5, 10], device=device)
        # 高压差-Ki表
        self.PID_Ki_list = torch.tensor([0.1503906, 0.1503906, 0.1503906, 0.1503906, 0.1503906, 0.1503906], device=device)
        self.diffpress_Ki_list = torch.tensor([-10, -2.5, -1, 1, 2.5, 10], device=device)

        # 插值表
        self.lo_press_table = torch.tensor([100.0, 150, 200, 250, 300, 350, 400, 450, 500, 550], device=device)
        self.high_press_table = torch.tensor([200.0, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000], device=device)

        # 第二个维度(高压压力)：200,400,600,800,1000,1200,1400,1600,1800,2000                     第一个维度(低压压力)
        self.com_speed_min_table_modena = torch.tensor([[2000.0, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000],  # 100
                                            [1600, 1600, 1600, 1600, 1600, 1700, 1800, 1900, 2000, 2000],  # 150
                                            [1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1600, 2000],  # 200
                                            [900, 900, 950, 1000, 1050, 1100, 1150, 1200, 1600, 2000],  # 250
                                            [800, 800, 800, 800, 900, 1000, 1100, 1200, 1600, 2000],  # 300
                                            [800, 800, 800, 800, 800, 900, 1050, 1200, 1600, 2000],  # 350
                                            [800, 800, 800, 800, 800, 800, 1000, 1200, 1600, 2000],  # 400
                                            [800, 800, 800, 800, 800, 800, 950, 1200, 1600, 2000],  # 450
                                            [800, 800, 800, 800, 800, 800, 900, 1200, 1600, 2000],  # 500
                                            [800, 800, 800, 800, 800, 800, 850, 1200, 1600, 2000]], device=device)  # 550

        # 第二个维度(高压压力)：200,400,600,800,1000,1200,1400,1600,1800,2000                     第一个维度(低压压力)
        self.com_speed_min_table_lemans_1_2 = torch.tensor([[1000.0, 1000, 1000, 1000, 1000, 1700, 2400, 3000, 3000, 3000],  # 100
                                            [900, 900, 900, 900, 900, 950, 1500, 2000, 2500, 3000],  # 150
                                            [800, 800, 800, 800, 800, 850, 900, 950, 1000, 1500],  # 200
                                            [800, 800, 800, 800, 800, 800, 850, 900, 1000, 1400],  # 250
                                            [800, 800, 800, 800, 800, 800, 800, 900, 1000, 1300],  # 300
                                            [800, 800, 800, 800, 800, 800, 800, 900, 1000, 1300],  # 350
                                            [800, 800, 800, 800, 800, 800, 800, 900, 1000, 1300],  # 400
                                            [800, 800, 800, 800, 800, 800, 800, 900, 1000, 1350],  # 450
                                            [800, 800, 800, 800, 800, 800, 800, 900, 1000, 1400],  # 500
                                            [800, 800, 800, 800, 800, 800, 800, 900, 1000, 1450]], device=device)  # 550

        # 第二个维度(高压压力)：200,400,600,800,1000,1200,1400,1600,1800,2000                     第一个维度(低压压力)
        self.com_speed_min_table_lemans_5 = torch.tensor([[2000.0, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000],  # 100
                                            [1600, 1600, 1600, 1600, 1600, 1700, 1800, 1900, 2000, 2000],  # 150
                                            [1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1600, 2000],  # 200
                                            [900, 900, 950, 1000, 1050, 1100, 1150, 1200, 1600, 2000],  # 250
                                            [800, 800, 800, 800, 900, 1000, 1100, 1200, 1600, 2000],  # 300
                                            [800, 800, 800, 800, 800, 900, 1050, 1200, 1600, 2000],  # 350
                                            [800, 800, 800, 800, 800, 800, 1000, 1200, 1600, 2000],  # 400
                                            [800, 800, 800, 800, 800, 800, 950, 1200, 1600, 2000],  # 450
                                            [800, 800, 800, 800, 800, 800, 900, 1200, 1600, 2000],  # 500
                                            [800, 800, 800, 800, 800, 800, 850, 1200, 1600, 2000]], device=device)  # 550

        self.Kd_paras = torch.tensor([0.048828], device=device)

        self.Ivalue = torch.tensor([0.0], device=device)
        # self.Ivalue = None

        # 蒸发器温度

        self.last_Pout = torch.tensor([0.0], device=device)
        self.last_Dout = torch.tensor([0.0], device=device)

        self.last_diff_hp = 0

        self.K_high_pressure = 0.0244
        self.K_low_pressure = 0.0996

        self.offset_change_max = 800.0
        self.offset_change_min = -800.0
        self.last_offset = 0
        # self.last_ac_pid_out_hp =800.0

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

        # 蒸发器温度
        temp_evap = x[:,15]

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
        diff_lo_pressure = (aim_lo_pressure - lo_pressure) * self.K_high_pressure

        # diff_hi_pressure = torch.min(diff_hi_pressure, diff_lo_pressure)


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
            Kp = inter1D(self.diffpress_Kp_list, self.PID_Kp_list, diff_hi_pressure)*AC_KpRate
            Ki = inter1D(self.diffpress_Ki_list, self.PID_Ki_list, diff_hi_pressure)*AC_KpRate
            Kd = self.Kd_paras*AC_KpRate
        except:
            print(diff_hi_pressure.shape)

        if 'modena' in self.car_type:
            com_speed_min_table = self.com_speed_min_table_modena
        elif 'lemans_1' in self.car_type or 'lemans_2' in self.car_type:
            com_speed_min_table = self.com_speed_min_table_lemans_1_2
        elif 'lemans_5' in self.car_type:
            com_speed_min_table = self.com_speed_min_table_lemans_5
        else:
            com_speed_min_table = self.com_speed_min_table_lemans_5

        com_speed_min = inter2D(self.lo_press_table, self.high_press_table, com_speed_min_table, lo_pressure, hi_pressure)


        # if self.Ivalue == None:
        #     self.Ivalue = Ki * diff_hi_pressure
        last_Ivalue = self.Ivalue
        if abs(diff_hi_pressure) < 0.39063:
            self.Ivalue = torch.zeros_like(self.Ivalue) * diff_hi_pressure
        else:
            # self.Ivalue = (self.Ivalue + diff_hi_pressure)/2
            self.Ivalue = (self.Ivalue + diff_hi_pressure)/2

        # self.Ivalue = (self.Ivalue + diff_hi_pressure_real) / 2
        # if len(last_ac_pid_out_hp.shape) == 1:
        #     last_ac_pid_out_hp = last_ac_pid_out_hp.unsqueeze(1)
        last_ac_pid_out_hp = last_ac_pid_out_hp.view(-1, 1)
        Iout = Ki * (self.Ivalue + last_Ivalue)/2
        # Iout = ac_pid_iout_hp
        # Pout = Kp * diff_hi_pressure
        Pout = Kp * diff_hi_pressure
        # Pout = ac_pid_pout_hp

        Dout = 0.0
        Diffout = (Pout + Dout) - (self.last_Pout + self.last_Dout)
        offset = (Iout + Diffout) * 85

        # ac_pid_out_hp = last_ac_pid_out_hp + offset
        # ac_pid_out_hp = torch.clamp(ac_pid_out_hp, min=com_speed_min.item())
        # offset = ac_pid_out_hp - last_ac_pid_out_hp



            # 保护逻辑
        f_min, R_min = get_acpid_discount(offset, hi_pressure, temp_p_h_2, raito_press, lo_pressure)
        if R_min != -1.0 and R_min < 1.0:
            R_list = [0, 0.40, 0.60, 0.80, 1.00]
            com_speed_list = [500, 100, 10, 5, 1]
            max_change = inter1D(R_list, com_speed_list, R_min)
            offset = torch.clamp(offset, min=-max_change, max=max_change)

        if f_min != -1.0 and f_min < 1.0:
            f = f_min
            offset = torch.clamp(offset, max=0.0)
        else:
            f = 1.0
            offset = torch.clamp(offset, max=800.0)



        ac_pid_out_hp = (last_ac_pid_out_hp + offset) * f
        ac_pid_out_hp = torch.clamp(ac_pid_out_hp,min=com_speed_min.item(), max=torch.max(last_ac_pid_out_hp + 800, torch.tensor(1700.0)).item())
        diff_ac_pid_out = ac_pid_out_hp - last_ac_pid_out_hp

        # 变化逻辑
        if last_ac_pid_out_hp + 100 > com_speed_min.item():
            # 蒸发器温度查表
            temp_evap_up_max = inter1D(self.EvapTemp_table[:, 0], self.EvapTemp_table[:, 1], temp_evap)
            # 最大上升速度
            max_up_change_spd = min([torch.tensor(200), diff_ac_pid_out, AC_KpRate*78, temp_evap_up_max])
            # 最大下降速度
            max_down_change_spd = -200
        else:
            max_up_change_spd = torch.tensor(8500.0)
            max_down_change_spd = -8500

        offset = torch.clamp(offset, min=max_down_change_spd, max=max_up_change_spd.item())

        ac_pid_out_hp = last_ac_pid_out_hp + offset

        # 上下限
        # max_ac_pid_out_hp = torch.max(last_ac_pid_out_hp + max_change_up,torch.tensor(1700.0))
        # max_ac_pid_out_hp = torch.min(max_ac_pid_out_hp,torch.tensor(8000.0))

        if ac_pid_out_hp < com_speed_min.item() - 120:
            com_speed_min = torch.tensor([800.0])

        # ac_pid_out_hp = last_ac_pid_out_hp + offset
        ac_pid_out_hp = torch.clamp(ac_pid_out_hp, min=com_speed_min.item(), max=8000.0)

        # ac_pid_out_hp = self.k*ac_pid_out_hp + self.bais

        self.last_diff_hp = diff_hi_pressure
        self.last_Pout = Pout
        self.last_Dout = Dout
        self.last_offset = ac_pid_out_hp.detach() - last_ac_pid_out_hp
        self.last_ac_pid_out_hp = ac_pid_out_hp.detach()

        if self.last_offset >= 800:
            self.last_offset = 0.0
            self.last_ac_pid_out_hp = last_ac_pid_out_hp.detach()
            return last_ac_pid_out_hp


        return ac_pid_out_hp

if __name__ == '__main__':
    offset,f = get_acpid_discount(900, 1000, 105, 1, 1000)
    print(offset,f)