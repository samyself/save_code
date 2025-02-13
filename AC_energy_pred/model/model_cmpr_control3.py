import torch
import torch.nn as nn
import torch.nn.init as init
from torch.onnx.symbolic_opset9 import tensor


def init_weights_xavier_uniform(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

class MLPModel1(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[256, 128, 64, 32, 16], dropout_val=[0.5,0.5]):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim[0])

        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        # self.linear4 = nn.Linear(hidden_dim[2], hidden_dim[3])
        # self.linear5 = nn.Linear(hidden_dim[3], hidden_dim[4])
        self.out_linear = nn.Linear(hidden_dim[2], output_dim)
        self.dp1 = nn.Dropout(dropout_val[0])
        self.dp2 = nn.Dropout(dropout_val[1])

        self.bn1 = torch.nn.BatchNorm1d(hidden_dim[0])
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim[1])
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim[2])
        # self.bn4 = torch.nn.BatchNorm1d(hidden_dim[3])
        # self.bn5 = torch.nn.BatchNorm1d(hidden_dim[4])

        self.relu  = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        #第一层
        x1 = self.linear1(x)
        x1 = self.bn1(x1)
        x1 = self.sigmoid(x1)
        x1 = self.dp1(x1)

        # # 第二层
        x2 = self.linear2(x1)
        x2 = self.bn2(x2)
        x2 = self.sigmoid(x2)
        x2 = self.dp1(x2)

        # # 第三层
        x3 = self.linear3(x2)
        x3 = self.bn3(x3)
        x3 = self.sigmoid(x3)
        x3 = self.dp1(x3)


        out = self.out_linear(x3)
        out = self.relu(out)

        return out


def norm(tensor,max,min):
    output = (tensor - min)/(max - min)
    return output



class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[2048, 2048, 2048], dropout_val=[0,0.5]):
        super().__init__()
        self.mlp1 = MLPModel1(6,1,hidden_dim=hidden_dim)
        # self.mlp2 = MLPModel1(1, 1,hidden_dim=hidden_dim)
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

        # 流量系数*流通截面积*流量密度
        # self.Cd_A0_row = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.cab_pos_k = nn.Parameter(torch.zeros(1), requires_grad=True)
        # self.cab_pos_n = nn.Parameter(torch.ones(1), requires_grad=True)
        # self.cab_pos_bias = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.Cd_A0_row = nn.Parameter(torch.rand(1), requires_grad=True)
        self.cab_pos_k = nn.Parameter(torch.rand(1), requires_grad=True)
        self.cab_pos_n = nn.Parameter(torch.ones(1), requires_grad=True)
        self.cab_pos_bias = nn.Sequential(nn.Linear(6,16),nn.ReLU(),
                                          nn.Linear(16,16),nn.ReLU(),
                                          nn.Linear(16,1))



    def forward(self, x):
        # 压缩机排气温度、内冷温度、饱和高压、压缩机进气温度、饱和低压、目标饱和高压、目标过冷度、目标过热度
        # 压缩机转速、膨胀阀开度

        # 输出限制
        # 压缩机排气温度
        temp_p_h_2 = x[:,0]
        # temp_p_h_2 = torch.clamp(x[:,0],max=self.temp_p_h_2_max,min=self.temp_p_h_2_min)
        # temp_p_h_2 = norm(temp_p_h_2,max=self.temp_p_h_2_max,min=self.temp_p_h_2_min)
        # 内冷温度
        temp_p_h_5 = x[:,1]
        # temp_p_h_5 = torch.clamp(x[:,1],max=self.temp_p_h_5_max,min=self.temp_p_h_5_min)
        # temp_p_h_5 = norm(temp_p_h_5,max=self.temp_p_h_5_max,min=self.temp_p_h_5_min)
        # 饱和高压
        hi_pressure = x[:,2]
        # hi_pressure = torch.clamp(x[:,2],max=self.hi_pressure_max,min=self.hi_pressure_min)
        # hi_pressure = norm(hi_pressure,max=self.hi_pressure_max,min=self.hi_pressure_min)
        # 压缩机进气温度
        temp_p_h_1_cab_heating = x[:,3]
        # temp_p_h_1_cab_heating = torch.clamp(x[:,3],max=self.temp_p_h_1_cab_heating_max,min=self.temp_p_h_1_cab_heating_min)
        # temp_p_h_1_cab_heating = norm(temp_p_h_1_cab_heating,max=self.temp_p_h_1_cab_heating_max,min=self.temp_p_h_1_cab_heating_min)
        # 饱和低压
        lo_pressure = x[:,4]
        # lo_pressure = torch.clamp(x[:,4],max=self.lo_pressure_max,min=self.lo_pressure_min)
        # lo_pressure = norm(lo_pressure,max=self.lo_pressure_max,min=self.lo_pressure_min)
        # 目标饱和高压
        aim_hi_pressure = x[:,5]
        # aim_hi_pressure = torch.clamp(x[:,5],max=self.aim_hi_pressure_max,min=self.aim_hi_pressure_min)
        # aim_hi_pressure = norm(aim_hi_pressure,max=self.aim_hi_pressure_max,min=self.aim_hi_pressure_min)
        # 目标饱和低压
        # aim_lo_pressure = torch.clamp(x[:,6],max=700.0,min=100.0)
        # aim_lo_pressure = torch.clamp(x[:,6],max=700.0)
        # aim_lo_pressure = (aim_lo_pressure - 100.0) / (700.0 - 100.0)
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


        # 进气 压缩机温度差 内冷 饱高 压缩比 包和高压差
        use_x1 = torch.cat((temp_p_h_2.unsqueeze(1),dif_temp_p_h.unsqueeze(1),temp_p_h_5.unsqueeze(1),
                            hi_pressure.unsqueeze(1),rate_pressure.unsqueeze(1),dif_hi_pressure.unsqueeze(1)), dim=1)
        compressor_speed = self.mlp1(use_x1)

        # 进气 压缩机温度差 内冷 饱高 压缩比 包和高压差
        use_x2 = torch.cat((temp_p_h_2.unsqueeze(1),dif_temp_p_h.unsqueeze(1),temp_p_h_5.unsqueeze(1),
                            hi_pressure.unsqueeze(1),rate_pressure.unsqueeze(1),dif_hi_pressure.unsqueeze(1)), dim=1)
        m_val = (torch.clamp(self.Cd_A0_row*(hi_pressure - lo_pressure),min=0.0))**(1/2)
        cab_heating_status_act_pos = self.cab_pos_k*(m_val**self.cab_pos_n) + self.cab_pos_bias(use_x2)
        # 结果输出限幅
        if not self.training:
            # compressor_speed = torch.round(compressor_speed).int()
            cab_heating_status_act_pos = torch.round(cab_heating_status_act_pos).int()
            compressor_speed = torch.clamp(compressor_speed,max=self.compressor_speed_max,min=self.compressor_speed_min)
            cab_heating_status_act_pos = torch.clamp(cab_heating_status_act_pos,max=self.cab_heating_status_act_pos_max,min=self.cab_heating_status_act_pos_min)
        return compressor_speed,cab_heating_status_act_pos

