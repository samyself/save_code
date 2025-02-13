import torch
import torch.nn as nn
import torch.nn.init as init

# 温度vs饱和压力转换
def tem_sat_press(press=None, tem=None):
    # 输出tem：摄氏度
    # 输出prss：MPa
    if press is not None and tem is None:
        mode = 1
        val = press
    elif tem is not None and press is None:
        mode = 0
        val = tem
    else:
        print("error")
        return None
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
    val_idx = torch.searchsorted(tem_sat_press[:, mode].contiguous(), val) - 1
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


# 单线性插值
def inter1D(x_list, y_list, x):
    if not isinstance(x_list, torch.Tensor):
        x_list = torch.tensor(x_list)
    if not isinstance(y_list, torch.Tensor):
        y_list = torch.tensor(y_list)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    # 直接将x中的最大值与最小值作为x_list的最小、大边界
    x_min = torch.min(x)
    x_max = torch.max(x)
    if x_min<x_list[0]:
        y_left = (y_list[1] - y_list[0]) / (x_list[1] - x_list[0]) * (x_min - x_list[0]) + y_list[0]
        y_list[0] = y_left
        x_list[0] = x_min
    if x_max>x_list[-1]:
        y_right = (y_list[-1] - y_list[-2]) / (x_list[-1] - x_list[-2]) * (x_max - x_list[-1]) + y_list[-1]
        y_list[-1] = y_right
        x_list[-1] = x_max

        # 确保输入张量是连续的
    x = x.contiguous()

    # 找到输入低压和高压在表中的位置
    x_index = searchsorted(x_list, x) - 1

    y = y_list[x_index] + (x - x_list[x_index]) * (y_list[x_index + 1] - y_list[x_index]) / (x_list[x_index + 1] - x_list[x_index])
    return y


def searchsorted(sorted_sequence, values, out_int32=False, right=False):
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

    indices = torch.zeros_like(values)
    for i,value in enumerate(values):
        left, right_bound = 0, len(sorted_sequence)

        while left < right_bound:
            mid = (left + right_bound) // 2
            if (sorted_sequence[mid] < value) or (right and sorted_sequence[mid] <= value):
                left = mid + 1
            else:
                right_bound = mid

        indices[i] = left

    indices_tensor = torch.tensor(indices, dtype=torch.int32 if out_int32 else torch.int64)
    return indices_tensor


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
        self.CP_PID_Ki_list = torch.tensor(
            [0.0500488, 0.0200195, 0.0080566, 0.0024414, 0.0024414, 0.0080566, 0.0200195, 0.0500488])
        self.CP_diffpress_Ki_list = torch.tensor([-10, -5, -2, -0.75, 0.75, 2, 5, 10])

        # 转速初值表
        self.CP_InitValue_list = torch.tensor([30, 35, 40, 45, 50, 55, 60, 65])
        self.CP_com_sped_list = torch.tensor([0, 2000, 3000, 4000, 5000, 6000, 7000, 8000])

        self.Delta = None


        self.MLP1 = MLP(input_size=4, hidden_sizes=[256, 256], output_size=1, use_batchnorm=False, use_dropout=False)
        self.MLP2 = MLP(input_size=1, hidden_sizes=[256, 256], output_size=1, use_batchnorm=False, use_dropout=False)

        init_weights_xavier_uniform(self.MLP1)
        init_weights_xavier_uniform(self.MLP2)
        self.k1 = nn.Parameter(torch.randn([1]), requires_grad=True)

    def forward(self, x):
        torch.autograd.set_detect_anomaly(True)
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
        # 上一时刻exv_oh_pid
        last_exv_oh_pid = x[:, 7]

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

        # """
        #     开度前向传播
        # """
        try :
            Kp = inter1D(self.CP_diffpress_Kp_list, self.CP_PID_Kp_list, SCErr)
            Ki = inter1D(self.CP_diffpress_Ki_list, self.CP_PID_Ki_list, SCErr)
            Kd = 0
        except:
            print('SCErr = ', SCErr)
        if SCErr < 0 or SCRaw < 0 or last_exv_oh_pid >= 38 or SCRaw != 0:
            Ki = 0
        else:
            pass

        if self.Delta == None:
            self.Delta = inter1D(self.CP_com_sped_list, self.CP_InitValue_list, compressor_speed_last)

        self.Delta = (self.Delta +  Ki * SCErr)
        offset = Kp * SCErr + self.Delta

        use_x = torch.cat((temp_p_h_5.unsqueeze(1), hi_pressure.unsqueeze(1),
                           temp_p_h_1_cab_heating.unsqueeze(1), lo_pressure.unsqueeze(1)), dim=1)

        part1 = self.MLP1(use_x)


        # part2 = self.MLP2(compressor_speed_last.unsqueeze(1))

        exv_oh_pid = last_exv_oh_pid.unsqueeze(1) + (offset + part1) * self.k1
        # exv_oh_pid = offset


        exv_oh_pid = torch.relu(exv_oh_pid - 12.0) + 12.0
        exv_oh_pid = 100.0 - torch.relu(100.0 - exv_oh_pid)

        return exv_oh_pid


if __name__ == '__main__':
    print('1')
    # 1

