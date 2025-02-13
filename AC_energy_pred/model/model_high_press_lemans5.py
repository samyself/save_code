import torch
import torch.nn as nn
import torch.nn.init as init
from AC_energy_pred.model.data_utils_common import inter2D, inter1D, searchsorted


def my_relu(input_tensor):
    # 创建一个与输入张量形状相同的全零张量
    zero_tensor = torch.zeros_like(input_tensor)
    # 应用 max 函数，比较每个元素并返回较大的那个
    return torch.max(zero_tensor, input_tensor)

def tem_sat_press(val):
    # 输入压力
    # 输出tem：摄氏度

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
    val_idx = searchsorted(tem_sat_press[:, 1], val) - 1
    # 确保索引在有效范围内
    val_idx = torch.clamp(val_idx, 0, tem_sat_press.shape[0] - 2)

    output1 = tem_sat_press[val_idx, 0]

    output2 = tem_sat_press[val_idx + 1, 0]

    val_w1 = tem_sat_press[val_idx, 1]
    val_w2 = tem_sat_press[val_idx + 1, 1]

    w = (val - val_w1) / (val_w2 - val_w1)
    output = w * (output2 - output1) + output1
    return output


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
        # self.k1 = nn.Parameter(torch.tensor(0.22743825614452362), requires_grad=False)
        # self.k2 = nn.Parameter(torch.tensor(0.7088953852653503), requires_grad=False)
        # self.k3 = nn.Parameter(torch.tensor(1.0006617307662964), requires_grad=False)
        # self.k4 = nn.Parameter(torch.tensor(0.9992321729660034), requires_grad=False)
        # self.k5 = nn.Parameter(torch.tensor(-0.021352600306272507), requires_grad=False)
        self.MLP1 = MLP(input_size=1, output_size=1, hidden_sizes=[64, 64], use_batchnorm=True, use_dropout=False)
        self.MLP2 = MLP(input_size=8, output_size=1, hidden_sizes=[64, 64], use_batchnorm=True, use_dropout=False)
        # self.t1 = MLP(input_size=14, output_size=1, hidden_sizes=[64, 64], use_batchnorm=True, use_dropout=False)
        # self.t2 = MLP(input_size=14, output_size=1, hidden_sizes=[64, 64], use_batchnorm=True, use_dropout=False)

        init_weights_xavier_uniform(self.MLP1)
        init_weights_xavier_uniform(self.MLP2)
        # init_weights_xavier_uniform(self.t1)
        # init_weights_xavier_uniform(self.t2)
        self.a0 = nn.Parameter(torch.randn(1,1), requires_grad=True)
        # self.a1 = nn.Parameter(torch.randn(1, 1), requires_grad=True)
        # self.a2 = nn.Parameter(torch.randn(1, 1), requires_grad=True)
        # self.a3 = nn.Parameter(torch.randn(1, 1), requires_grad=True)
        # self.a4 = nn.Parameter(torch.randn(1, 1), requires_grad=True)

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
        temp_p_h_1_cab_heating = x[:, 4]
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

        #压缩比
        ratio = (hi_pressure[:, 0].detach() / lo_pressure)
        ratio = torch.clamp(ratio, 3, 100)
        # 风温对冷却流量的影响
        wind_and_coolant = air_temp_before_heat_exchange * heat_coolant_vol

        # 饱和温度
        temp_high_press = tem_sat_press(hi_pressure[:,0].detach())

        # model_pys_out = temp_p_h_1_cab_heating + (
        #             self.k1 * (hi_pressure ** self.k2 ) * (
        #                             compressor_speed ** self.k3)) - 273.15

        use_x2 = torch.cat((hi_pressure.detach(), compressor_speed.unsqueeze(1), temp_p_h_1_cab_heating.unsqueeze(1),
                            lo_pressure.unsqueeze(1), cab_heating_status_act_pos.unsqueeze(1),
                            temp_p_h_5.detach(),ratio.unsqueeze(1),wind_and_coolant.unsqueeze(1),), dim=1)
        #
        # use_t = torch.cat((x[:,[0,1,2,3,4,5,6,7,9]],ratio.unsqueeze(1),hi_pressure.detach(),temp_p_h_5.detach(),temp_higg_press.unsqueeze(1),wind_and_coolant.unsqueeze(1)), dim=1)

        part1 = self.MLP1(hi_pressure)
        # part1 = self.k1 * hi_pressure
        part2 = self.MLP2(use_x2)
        out =  part1 + part2 + (self.a0[0,0]* temp_high_press).reshape(-1,1)
        # out = torch.clamp(out, temp_p_h_1_cab_heating, temp_higg_press)
        # out = torch.clamp(out, 0, 100)
        return out

class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[256, 128, 16], dropout_val=[0, 0]):
        super().__init__()

        self.linear1 = torch.nn.Linear(8, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 16)
        self.linear4 = torch.nn.Linear(16, 3)
        self.bn0 = nn.BatchNorm1d(8)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(16)
        # self.dp1 = nn.Dropout(0.05)
        # self.dp2 = nn.Dropout(0.02)
        self.relu = my_relu
        # self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.k1 = nn.Parameter(torch.tensor(0.99))
        self.k2 = nn.Parameter(torch.tensor(1.01))
        self.k3 = nn.Parameter(torch.tensor(1.1))
        self.k4 = nn.Parameter(torch.tensor(1.0))
        self.k5 = nn.Parameter(torch.tensor(0.9))

    def forward(self, x):
        x1 = self.bn0(x)
        h1 = self.relu(self.bn1(self.linear1(x1)))
        # h1 = self.dp1(h1)
        h2 = self.relu(self.bn2(self.linear2(h1)))
        # h2 = self.dp2(h2)
        h3 = self.relu(self.bn3(self.linear3(h2)))
        out = self.linear4(h3)
        return out


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Cool_Temp_and_Hi_Press_model = MLPModel(8, 2, hidden_dim=[256, 128, 16], dropout_val=[0, 0])

        self.Com_Out_Temp_model = Com_Out_Temp_model()
        init_weights_xavier_uniform(self.Cool_Temp_and_Hi_Press_model)
        init_weights_xavier_uniform(self.Com_Out_Temp_model)

        # 冻结 Cool_Temp_and_Hi_Press_model 的参数
        for param in self.Cool_Temp_and_Hi_Press_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x1 = x[:, [0, 1, 2, 3, 4, 5, 7, 9]]
        Cool_Temp_and_Hi_Press = self.Cool_Temp_and_Hi_Press_model(x1)
        temp_p_h_5 = Cool_Temp_and_Hi_Press[:, 0:1]
        hi_pressure = Cool_Temp_and_Hi_Press[:, 2:3]

        temp_p_h_5 = torch.clamp(temp_p_h_5, 0)
        hi_pressure = torch.clamp(hi_pressure, 0)


        Com_Out_Temp = self.Com_Out_Temp_model(x, hi_pressure, temp_p_h_5)
        # Com_Out_Temp = x[:, 0:1]
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
