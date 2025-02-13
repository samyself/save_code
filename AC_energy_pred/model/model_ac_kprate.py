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


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ac_pid_out_hp_max = 8500.0
        self.ac_pid_out_hp_min = 800.0
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

        #设定温度
        self.temp_set = torch.tensor([18.0,20,22,24,26,28,30,31.5,32])
        #环境温度
        self.temp_envr = torch.tensor([-30.0,-20,-10,0,5,10,15,20,25,30,35,40,45,50])
                                     # 第二个维度(环境温度)：-30.0,-20,-10,0,5,10,15,20,25,30,35,40,45,50    第一个维度(设定温度)
        self.CabinSP_table = torch.tensor([[17.0, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17],               # 18
                                       [20, 20, 19.5, 19.5, 19.5, 19, 19, 19, 18.5, 18.5, 18, 18, 18, 18],     # 20
                                        [22, 22, 22, 22.5, 22.5, 22.5, 22, 22, 21, 21, 21, 21, 20.5, 20],      # 22
                                        [24, 24.5, 25.5, 25.5, 26, 26, 25.5, 25, 24.5, 24, 23.5, 23, 23, 23],  # 24
                                        [27, 26.5, 27, 27.5, 28, 28, 27.5, 27, 26.5, 26, 25.5, 26, 26, 26],    # 26
                                        [29, 28.5, 28.5, 29.5, 30, 30, 29.5, 29, 29, 29, 28, 28, 29, 29],      # 28
                                        [31, 30.5, 30.5, 31.5, 32, 32, 32, 31, 31, 31, 31, 31, 31, 31],        # 30
                                        [32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 32, 32],              # 31.5
                                        [32, 32, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36]])            # 32

        # 插值表
        self.lo_press_table = torch.tensor([100.0, 150, 200, 250, 300, 350, 400, 450, 500, 550])
        self.high_press_table = torch.tensor([200.0, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])

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
                                            [800, 800, 800, 800, 800, 800, 850, 1200, 1600, 2000]])  # 550

        self.MLP1 = MLP(input_size=4, hidden_sizes=[512, 512], output_size=1,use_batchnorm=True,use_dropout=False)

        init_weights_xavier_uniform(self.MLP1)

        # 高压差-Kp表
        self.PID_Kp_list = torch.tensor([0.5, 0.3417968, 0.3417968, 0.3417968, 0.3417968,0.5])
        self.diffpress_Kp_list = torch.tensor([-10, -2.5, -1, 1, 2.5, 10])
        # 高压差-Ki表
        self.PID_Ki_list = torch.tensor([0.1503906, 0.1503906, 0.1503906, 0.1503906, 0.1503906, 0.1503906])
        self.diffpress_Ki_list = torch.tensor([-10, -2.5, -1, 1, 2.5, 10])
        self.Kd_paras = torch.tensor([0.0097656])

        self.last_ac_pid_out_hp = None

    def forward(self, x):
        # 压缩机排气温度、内冷温度、饱和高压、压缩机进气温度、饱和低压、目标饱和高压、目标过冷度、目标过热度
        # 压缩机转速、膨胀阀开度
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        # 输出限制
        # 上一时刻ac_pid_pout
        if not self.training:
            last_ac_pid_out_hp = self.last_ac_pid_out_hp
        else:
            last_ac_pid_out_hp = x[:,0]
        # 当前环境温度
        temp_amb = x[:, 1]

        # 当前主驾驶设定温度
        cab_fl_set_temp = x[:,2]
        # 当前副驾驶设定温度
        cab_fr_set_temp = x[:,3]
        # 上一时刻饱和低压
        lo_pressure = x[:,4]
        # 上一时刻饱和高压
        hi_pressure = x[:,5]
        # 目标饱和低压
        aim_lo_pressure = x[:,6]
        # 目标饱和高压
        aim_hi_pressure = x[:,7]

        # 当前乘务舱温度
        temp_incar = x[:,8]
        # 电池请求冷却液温度
        temp_battery_req = x[:,9]
        # 冷却液进入电池的温度
        temp_coolant_battery_in = x[:,10]

        # ac_kp_rate_last
        ac_kp_rate_last = x[:,11]

        # 压缩机排气温度
        temp_p_h_2 = x[:,12]
        # 压缩机进气温度
        temp_p_h_1_cab_heating = x[:,13]
        # 内冷温度
        temp_p_h_5 = x[:,14]

        # 高压偏差
        diff_hi_pressure = (hi_pressure - aim_hi_pressure)

        com_speed_min = inter2D(self.lo_press_table, self.high_press_table, self.com_speed_min_table, lo_pressure,
                                hi_pressure)

        #   DVT_CabinErr_FL = f(主驾设定温度, 环境温度) 查表 - SEN_Incar(乘员舱温度)
        DVT_CabinErr_FL = inter2D(self.temp_set,self.temp_envr,self.CabinSP_table,cab_fl_set_temp,temp_amb) - temp_incar
        #   DVT_CabinErr_FR = f(副驾设定温度, 环境温度) 查表 - SEN_Incar(乘员舱温度)
        DVT_CabinErr_FR = inter2D(self.temp_set, self.temp_envr, self.CabinSP_table,cab_fr_set_temp, temp_amb) - temp_incar
        # DCT_DCT_ChillerTempErr = HP_Mode_BattWTReqOpt（电池请求冷却的温度） - SEN_CoolT_Battln（冷却液进入电池的温度）
        DCT_ChillerTempErr  = temp_battery_req - temp_coolant_battery_in

        use_x1 = torch.cat((DVT_CabinErr_FL.unsqueeze(1), DVT_CabinErr_FR.unsqueeze(1), temp_amb.unsqueeze(1), DCT_ChillerTempErr .unsqueeze(1)), dim=1)

        AC_KpRate = self.MLP1(use_x1)

        # ac_pid_out_hp = renorm(ac_pid_out_hp,self.ac_pid_out_hp_max,self.ac_pid_out_hp_min)

        return AC_KpRate

if __name__  == '__main__':

    temp_set = torch.tensor([18.0, 20, 22, 24, 26, 28, 30, 31.5, 32])
    # 环境温度
    temp_envr = torch.tensor([-30.0, -20, -10, 0, 5, 10, 15, 20, 25])


    print(inter1D(temp_set, temp_envr, torch.tensor([17.0,34])))