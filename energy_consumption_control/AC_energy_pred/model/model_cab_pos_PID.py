import torch
import torch.nn as nn
import torch.nn.init as init
from AC_energy_pred.model.model_format import inter1D,init_weights_xavier_uniform,norm,tem_sat_press

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
        self.CP_PID_Ki_list = torch.tensor([0.0500488, 0.0200195, 0.0080566, 0.0024414, 0.0024414, 0.0080566, 0.0200195, 0.0500488])
        self.CP_diffpress_Ki_list = torch.tensor([-10, -5, -2, -0.75, 0.75, 2, 5, 10])

        # 转速初值表
        self.CP_InitValue_list = torch.tensor([30,35,40,45,50,55,60,65])
        self.CP_com_sped_list = torch.tensor([0,2000,3000,4000,5000,6000,7000,8000])

        self.Delta = None
        self.last_cab_pos = 12.0

        self.k1 = nn.Parameter(torch.tensor([0.39]), requires_grad=True)
        self.b1 = MLP(input_size=8, hidden_sizes=[256,256], output_size=1, use_batchnorm=False, use_dropout=False)

        init_weights_xavier_uniform(self.b1)

    def forward(self, x):
        # 压缩机排气温度、内冷温度、饱和高压、压缩机进气温度、饱和低压、目标饱和高压、目标过冷度、目标过热度
        # 压缩机转速、膨胀阀开度
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        # 输出限制
        # 目标过冷度
        sc_tar_mode_10 = x[:, 0]
        # 目标过热度
        sh_tar_mode_10 = x[:, 1]
        # 压缩机进气温度
        temp_p_h_1_cab_heating = x[:, 2]
        # 内冷温度
        temp_p_h_5 = x[:, 3]
        # 饱和高压
        hi_pressure = x[:, 4]
        # 饱和低压
        lo_pressure = x[:, 5]
        # 压缩机转速
        compressor_speed_last = x[:, 6]

        # 实际过热度  = 压缩机进气温度 - 低压饱和压力对应温度
        sh_rel_mode_10 = temp_p_h_1_cab_heating - tem_sat_press(lo_pressure)
        # 实际过冷度 =  高压饱和压力对应温度 - 内冷温度
        sc_rel_mode_10 = tem_sat_press(hi_pressure) - temp_p_h_5
        # 过冷度偏差
        SCRaw = (sc_rel_mode_10 - sc_tar_mode_10)
        # 过热度偏差
        SCOffset = (sh_rel_mode_10 - sh_tar_mode_10)
        # SCErr
        SCErr = SCRaw + SCOffset



        """
            开度前向传播
        """
        # try :
        #     Kp = inter1D(self.CP_diffpress_Kp_list, self.CP_PID_Kp_list, SCErr)
        #     Ki = inter1D(self.CP_diffpress_Ki_list, self.CP_PID_Ki_list, SCErr)
        #     Kd = 0
        # except:
        #     print('SCErr = ', SCErr)
        # if SCErr < 0 or SCRaw < 0 or self.last_cab_pos >= 38 or SCRaw != 0:
        #     Ki = 0
        # else:
        #     pass
        #
        # if self.Delta == None:
        #     self.Delta = inter1D(self.CP_com_sped_list, self.CP_InitValue_list, compressor_speed_last)
        #
        #
        # self.Delta = (self.Delta +  Ki * SCErr)
        # offset = Kp * SCErr + self.Delta

        use_x = torch.cat((SCErr.unsqueeze(1),sh_rel_mode_10.unsqueeze(1),sc_rel_mode_10.unsqueeze(1),
                           temp_p_h_1_cab_heating.unsqueeze(1), temp_p_h_5.unsqueeze(1),
                            hi_pressure.unsqueeze(1), lo_pressure.unsqueeze(1), compressor_speed_last.unsqueeze(1)), dim=1)

        cab_pos = self.last_cab_pos  + self.b1(use_x)
        # print('cab_pos = ', cab_pos
        self.last_cab_pos = cab_pos.detach()
        # print('cab_pos = ', cab_pos)
        cab_pos = torch.clamp(cab_pos, 12, 100)


        return cab_pos


if __name__ == '__main__':
    print('1')
    # 1

