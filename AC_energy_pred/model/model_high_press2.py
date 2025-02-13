import torch
import torch.nn as nn
import torch.nn.init as init
import config


def init_weights_xavier_uniform(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
def searchsorted(sorted_sequence, values, out_int32: bool = False, right: bool = False) -> torch.LongTensor:
    """
    手动实现 torch.searchsorted 功能。

    参数:
        sorted_sequence (Tensor): 一个有序的一维张量。
        values (Tensor or Scalar): 要查找插入位置的值。
        out_int32 (bool, optional): 输出索引的类型是否为 int32，默认为 False（int64）。
        right (bool, optional): 是否使用右侧边界，默认为 False。

    返回:
        Tensor: 插入位置的索引。
    """
    if not isinstance(values, torch.Tensor):
        values = torch.tensor([values])
    if len(values.shape) == 2:
        values = values[:, 0]

    indices = torch.zeros_like(values)
    for i in range(values.shape[0]):
        left, right_bound = 0, len(sorted_sequence)
        value = values[i]
        while left < right_bound:
            mid = (left + right_bound) // 2
            if (sorted_sequence[mid] < value) or (right and sorted_sequence[mid] <= value):
                left = mid + 1
            else:
                right_bound = mid

        indices[i] = left

    indices = indices.to(torch.int32 if out_int32 else torch.int64)
    return indices
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
        outputs = []  # 用于存储每一层的输出
        for layer in self.network:
            x = layer(x)  # 逐层传递
            outputs.append(x)  # 保存每层的输出
        return outputs[-1]




class Com_Out_Temp_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lamuda1 = nn.Parameter(torch.tensor(3e2))
        self.lamuda2 = nn.Parameter(torch.tensor(5e2))
        self.k_t = nn.Parameter(torch.randn(1,1))
        self.k_k = nn.Parameter(torch.randn(1,1))


    def forward(self, x, hi_pressure, temp_p_h_5):
        # 蒸发器的风温
        air_temp_before_heat_exchange = x[:, 0] + 273.15
        # 蒸发器的风量
        wind_vol = x[:, 1]
        # 压缩机转速
        compressor_speed = x[:, 2]
        # CEXV膨胀阀开度
        cab_heating_status_act_pos = x[:, 3]
        # 压缩机进气温度
        temp_p_h_1_cab_heating = x[:, 4]
        # 饱和低压
        lo_pressure=x[:, 5]
        # 内冷温度
        # temp_p_h_5=x[:, 6]
        # 当前制热水泵流量
        heat_coolant_vol=x[:, 7]
        # 当前换热前的水温(暂无数据)
        cool_medium_temp_in=x[:, 8]
        # hvch出水温度
        hvch_cool_medium_temp_out=x[:, 9]
        #饱和高压

        # 管道长度和横截面积
        channel_len = config.channel_heating_place
        channel_s = config.channel_s
        #总时间
        t_all = channel_len / (wind_vol / channel_s)
        #气态到混合态换热时间
        t = t_all*self.k_t
        # t = torch.relu(t - t_all*0.1) + t_all*0.1
        # 饱和温度
        temp_high = tem_sat_press(hi_pressure) + 273.15
        # temp_high = torch.relu(temp_high)

        K0 = torch.clamp(self.lamuda1 / self.lamuda2,1e-4,1)
        # K1 = torch.exp(-(self.lamuda1 + self.lamuda2)*t)
        K1= 1 - self.k_k
        K1 = torch.clamp(K1,1e-4,1)
        C1 = (temp_high - air_temp_before_heat_exchange) / (K0 + K1)
        C2 = air_temp_before_heat_exchange + K1 * (temp_high - air_temp_before_heat_exchange) / (K0 + K1)
        out = C1 + C2 - 273.15
        return out

    def clip_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                with torch.no_grad():
                    layer.weight.clamp_(-self.max_value, self.max_value)
                    if layer.bias is not None:
                        layer.bias.clamp_(-self.max_value, self.max_value)





class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[256, 128, 16], dropout_val=[0,0]):
        super().__init__()

        self.linear1 = torch.nn.Linear(8, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128,16)
        self.linear4 = torch.nn.Linear(16, 3)
        self.bn0 = nn.BatchNorm1d(8)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(16)

        # self.dp1 = nn.Dropout(0.05)
        # self.dp2 = nn.Dropout(0.02)
        self.relu = nn.ReLU()
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
        # self.k1 = torch.tensor(0.2169)
        # self.k2 = torch.tensor(0.6979)
        # self.k3 = torch.tensor(1.0013)
        # self.k4 = torch.tensor(1.0002)
        # self.k5 = torch.tensor(-0.0461)

        self.Cool_Temp_and_Hi_Press_model = MLPModel(8, 2, hidden_dim=[256, 128, 16], dropout_val=[0,0])

        self.Com_Out_Temp_model = Com_Out_Temp_model()
        init_weights_xavier_uniform(self.Cool_Temp_and_Hi_Press_model)
        init_weights_xavier_uniform(self.Com_Out_Temp_model)

        # 冻结 Cool_Temp_and_Hi_Press_model 的参数
        for param in self.Cool_Temp_and_Hi_Press_model.parameters():
            param.requires_grad = False

    def forward(self, x, temp_p_h_5, hi_pressure):
        x1 = x[:,[0,1,2,3,4,5,7,9]]
        # Cool_Temp_and_Hi_Press = self.Cool_Temp_and_Hi_Press_model(x1)
        # temp_p_h_5 = Cool_Temp_and_Hi_Press[:, 0:1]
        # hi_pressure = Cool_Temp_and_Hi_Press[:, 2:3]
        #
        # temp_p_h_5 = torch.clamp(temp_p_h_5, 0)
        # hi_pressure = torch.clamp(hi_pressure, 0)

        Com_Out_Temp = self.Com_Out_Temp_model(x, hi_pressure, temp_p_h_5)
        if not ((Com_Out_Temp <= 1000) & (Com_Out_Temp >= -1000)).any() :
            print('Com_Out_Temp > 100')
        # if not self.training:
        #     Com_Out_Temp = torch.clamp(Com_Out_Temp, 40, 115)
        return torch.cat((temp_p_h_5, Com_Out_Temp,hi_pressure), dim=1)