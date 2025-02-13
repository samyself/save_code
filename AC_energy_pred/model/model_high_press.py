import torch
import torch.nn as nn
import torch.nn.init as init


def init_weights_xavier_uniform(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)


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


class Com_Out_Temp_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP1 = MLP(input_size=1, output_size=1, hidden_sizes=[512, 512], use_batchnorm=False, use_dropout=False)
        self.MLP2 = MLP(input_size=7, output_size=1, hidden_sizes=[512, 512], use_batchnorm=False, use_dropout=False)

        init_weights_xavier_uniform(self.MLP1)
        init_weights_xavier_uniform(self.MLP2)

    def forward(self, x, hi_pressure, temp_p_h_5):
        # 蒸发器的风温
        air_temp_before_heat_exchange = x[:, 0]
        # 蒸发器的风量
        wind_vol = x[:, 1]
        # 压缩机转速
        compressor_speed = x[:, 2]
        # CEXV膨胀阀开度
        cab_heating_status_act_pos = x[:, 3]
        # 压缩机进气温度
        temp_p_h_1_cab_heating = x[:, 4] + 273.15
        # 饱和低压
        lo_pressure = x[:, 5]
        # 内冷温度
        # temp_p_h_5=x[:, 6]
        # 当前制热水泵流量
        heat_coolant_vol = x[:, 7]
        # 当前换热前的水温(暂无数据)
        cool_medium_temp_in = x[:, 8]
        # hvch出水温度
        hvch_cool_medium_temp_out = x[:, 9]

        # 压缩比
        ratio = (hi_pressure[:, 0].detach() / lo_pressure)
        ratio = torch.clamp(ratio, 1, 100)
        # 风温对冷却流量的影响
        wind_and_coolant = air_temp_before_heat_exchange * heat_coolant_vol * hvch_cool_medium_temp_out

        use_x2 = torch.cat((compressor_speed.unsqueeze(1), temp_p_h_1_cab_heating.unsqueeze(1),
                            lo_pressure.unsqueeze(1), cab_heating_status_act_pos.unsqueeze(1),
                            temp_p_h_5, ratio.unsqueeze(1), wind_and_coolant.unsqueeze(1),), dim=1)

        part1 = self.MLP1(hi_pressure)
        part2 = self.MLP2(use_x2)
        out = part1 + part2
        return out


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 256)
        self.linear2 = torch.nn.Linear(256, 64)
        self.linear3 = torch.nn.Linear(64, 32)
        self.linear4 = torch.nn.Linear(32, 3)
        self.bn0 = nn.BatchNorm1d(10)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        # self.dp1 = nn.Dropout(0.02)
        # self.dp2 = nn.Dropout(0.02)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.k0 = nn.Parameter(torch.tensor(1.0))
        self.k6 = nn.Parameter(torch.tensor(1.0))
        self.k1 = nn.Parameter(torch.tensor(0.99))
        self.k2 = nn.Parameter(torch.tensor(1.01))
        self.k3 = nn.Parameter(torch.tensor(1.1))
        self.k4 = nn.Parameter(torch.tensor(1.0))
        self.k5 = nn.Parameter(torch.tensor(0.9))
        # self.k7 = nn.Parameter(torch.tensor(1.0))
        # self.b1 = nn.Parameter(torch.tensor(273.15))

    def forward(self, x):
        x0 = torch.cat((x[:, :6], x[:, 7].unsqueeze(1), x[:, 9:12]), dim=1).clone()
        x1 = self.bn0(x0)
        h1 = self.relu(self.bn1(self.linear1(x1)))
        # h1 = self.dp1(h1)
        h2 = self.relu(self.bn2(self.linear2(h1)))
        # h2 = self.dp2(h2)
        h3 = self.relu(self.bn3(self.linear3(h2)))
        out = self.linear4(h3)
        return out

# class MLPModel(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim=[256, 128, 16], dropout_val=[0, 0]):
#         super().__init__()
#
#         self.linear1 = torch.nn.Linear(8, 256)
#         self.linear2 = torch.nn.Linear(256, 128)
#         self.linear3 = torch.nn.Linear(128, 16)
#         self.linear4 = torch.nn.Linear(16, 3)
#         self.bn0 = nn.BatchNorm1d(8)
#         self.bn1 = nn.BatchNorm1d(256)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(16)
#         # self.dp1 = nn.Dropout(0.05)
#         # self.dp2 = nn.Dropout(0.02)
#         self.relu = nn.ReLU()
#         self.gelu = nn.GELU()
#         self.k1 = nn.Parameter(torch.tensor(0.99))
#         self.k2 = nn.Parameter(torch.tensor(1.01))
#         self.k3 = nn.Parameter(torch.tensor(1.1))
#         self.k4 = nn.Parameter(torch.tensor(1.0))
#         self.k5 = nn.Parameter(torch.tensor(0.9))
#
#     def forward(self, x):
#         x1 = self.bn0(x)
#         h1 = self.relu(self.bn1(self.linear1(x1)))
#         # h1 = self.dp1(h1)
#         h2 = self.relu(self.bn2(self.linear2(h1)))
#         # h2 = self.dp2(h2)
#         h3 = self.relu(self.bn3(self.linear3(h2)))
#         out = self.linear4(h3)
#         return out


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Cool_Temp_and_Hi_Press_model = MLPModel()

        self.Com_Out_Temp_model = Com_Out_Temp_model()
        init_weights_xavier_uniform(self.Cool_Temp_and_Hi_Press_model)
        init_weights_xavier_uniform(self.Com_Out_Temp_model)

        # 冻结 Cool_Temp_and_Hi_Press_model 的参数
        for param in self.Cool_Temp_and_Hi_Press_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x1 = x[:, [0, 1, 2, 3, 4, 5, 7, 9]]
        Cool_Temp_and_Hi_Press = self.Cool_Temp_and_Hi_Press_model(x)
        temp_p_h_5 = Cool_Temp_and_Hi_Press[:, 0:1]
        hi_pressure = Cool_Temp_and_Hi_Press[:, 2:3]

        temp_p_h_5 = torch.clamp(temp_p_h_5, 0)
        hi_pressure = torch.clamp(hi_pressure, 0)

        Com_Out_Temp = self.Com_Out_Temp_model(x, hi_pressure, temp_p_h_5)
        if not ((Com_Out_Temp <= 1000) & (Com_Out_Temp >= -1000)).any():
            print('Com_Out_Temp > 1000')

        return torch.cat((temp_p_h_5, Com_Out_Temp, hi_pressure), dim=1)


'''
    输出：
        # 饱和高压
        self.max_hisidep = 2400.0
        self.min_hisidep = 900.0
        # 压缩机排气温度
        self.max_cmdrdchat = 115.0
        self.min_cmdrdchat = 40.0
        # 内冷温度
        self.max_inrcondoutlT = 65.0
        self.min_inrcondoutlT = 20.0
    输入：
        # 压缩机转速
        self.max_cmpract = 8500.0 
        self.min_cmpract = 800.0
        # 膨胀阀开度
        self.max_cexv = 100.0
        self.min_cexv = 12.0
        # 压缩机进气温度
        self.max_cmdrdcint = 31.0
        self.min_cmdrdcint = -18.0
        # 饱和低压
        self.max_losidep = 700.0
        self.min_losidep = 110.0
        # 蒸发器进风温度
        self.max_air_temp_before_heat_exchange = 60.0
        self.min_air_temp_before_heat_exchange = -30.0
'''

