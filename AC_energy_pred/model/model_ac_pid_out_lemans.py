import torch
import torch.nn as nn
import torch.nn.init as init
import os
import sys

project_folder = os.path.abspath('..')
sys.path.append(os.path.join(project_folder, 'common'))
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


def init_weights_xavier_uniform(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)


def my_relu(input_tensor):
    # 创建一个与输入张量形状相同的全零张量
    zero_tensor = torch.zeros_like(input_tensor)
    # 应用 max 函数，比较每个元素并返回较大的那个
    return torch.max(zero_tensor, input_tensor)

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
        self.MLP = MLP(input_size=11, hidden_sizes=[256, 256], output_size=1, use_batchnorm=False)
        self.last_ac_pid_out_hp = None

        # 高压差-Kp表
        self.PID_Kp_list = torch.tensor([0.5, 0.3417968, 0.3417968, 0.3417968, 0.3417968, 0.5], device=device)
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

        self.AC_Kp_Rate = nn.Parameter(torch.rand(1, device=device), requires_grad=True)
        self.Kd_paras = torch.tensor([0.048828], device=device)

        self.Ivalue = torch.tensor([0.0], device=device)
        self.last_Pout = torch.tensor([0.0], device=device)
        self.last_Dout = torch.tensor([0.0], device=device)

        self.last_diff_hp = 0

        self.MLP1 = MLP(input_size=4, hidden_sizes=[512, 512], output_size=1, use_batchnorm=False, use_dropout=False)
        # self.MLP2 = MLP(input_size=11, hidden_sizes=[256, 256], output_size=1,use_batchnorm=False,use_dropout=True)

        init_weights_xavier_uniform(self.MLP1)
        # init_weights_xavier_uniform(self.MLP2)

        self.Iout_bais = MLP(input_size=4, hidden_sizes=[512, 512], output_size=1, use_batchnorm=False, use_dropout=False)
        self.Diffout_bais = MLP(input_size=4, hidden_sizes=[512, 512], output_size=1, use_batchnorm=False,
                                use_dropout=False)
        init_weights_xavier_uniform(self.Iout_bais)
        init_weights_xavier_uniform(self.Diffout_bais)

        self.MLP3 = MLP(input_size=6, hidden_sizes=[256, 256], output_size=1, use_batchnorm=False, use_dropout=False)
        self.MLP4 = MLP(input_size=2, hidden_sizes=[256, 256], output_size=1, use_batchnorm=False, use_dropout=False)

        init_weights_xavier_uniform(self.MLP3)
        init_weights_xavier_uniform(self.MLP4)

    def forward(self, x1):
        x1 = x1.view(-1, 7)
        x = x1.to(dtype=torch.float)

        last_ac_pid_out_hp = x[:, 0]
        last_ac_pid_out_hp = torch.clamp(last_ac_pid_out_hp, min=0.0, max=8000.0)
        # 上一时刻饱和低压
        lo_pressure = x[:, 1]
        lo_pressure = lo_pressure.clamp(min=0.0, max=1100.0)
        # 上一时刻饱和高压
        hi_pressure = x[:, 2]
        hi_pressure = hi_pressure.clamp(min=0.0, max=2400.0)
        # 目标饱和高压
        aim_hi_pressure = x[:, 3]
        aim_hi_pressure = aim_hi_pressure.clamp(min=0.0, max=2400.0)
        # 压缩机排气温度
        temp_p_h_2 = x[:, 4]
        temp_p_h_2 = temp_p_h_2.clamp(min=-100.0, max=200.0)
        # 压缩机进气温度
        temp_p_h_1_cab_heating = x[:, 5]
        temp_p_h_1_cab_heating = temp_p_h_1_cab_heating.clamp(min=-100.0, max=100.0)
        #  内冷温度
        temp_p_h_5 = x[:, 6]
        temp_p_h_5 = temp_p_h_5.clamp(min=-100.0, max=100.0)
        # 高压偏差
        diff_hi_pressure = (aim_hi_pressure - hi_pressure)
        # 压缩机温度差 = 压缩机排气温度 - 压缩机进气温度
        dif_temp_p_h = temp_p_h_2 - temp_p_h_1_cab_heating

        # com_speed_min = inter2D(self.lo_press_table.to(device), self.high_press_table.to(device), self.com_speed_min_table.to(device),
        #                         lo_pressure, hi_pressure).view(-1, 1)

        use_x1 = torch.cat((temp_p_h_2.unsqueeze(1), temp_p_h_5.unsqueeze(1), hi_pressure.unsqueeze(1),
                            temp_p_h_1_cab_heating.unsqueeze(1), lo_pressure.unsqueeze(1),
                            aim_hi_pressure.unsqueeze(1)), dim=1)

        Part1 = self.MLP3(use_x1)

        use_x2 = torch.cat((dif_temp_p_h.unsqueeze(1), diff_hi_pressure.unsqueeze(1)), dim=1)

        Part2 = self.MLP4(use_x2)

        ac_pid_out_hp = last_ac_pid_out_hp.view(-1, 1) + Part1 + Part2

        ac_pid_out_hp = torch.clamp(ac_pid_out_hp, min=800.0)

        return ac_pid_out_hp.view(-1, 1)
import random

if __name__ == '__main__':
    model = MyModel()
    model.eval()
    # random.seed(42)
    # matrix = [[round(random.uniform(-10000, 10000),4) for _ in range(7)] for _ in range(10000)]
    # matrix = torch.tensor(matrix, dtype=torch.float32)
    # matrix = torch.zeros((1, 7))
    matrix = torch.ones(1, 7) * 100000
    y = model(matrix)
    if y.isnan().any():
        print("存在NaN值")
    else:
        print("不存在NaN值")
    print(y.shape)