import torch
import torch.nn as nn
import torch.nn.init as init
from AC_energy_pred import config

from torch.onnx.symbolic_opset11 import unsqueeze
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

    indices = []
    for value in values:
        left, right_bound = 0, len(sorted_sequence)

        while left < right_bound:
            mid = (left + right_bound) // 2
            if (sorted_sequence[mid] < value) or (right and sorted_sequence[mid] <= value):
                left = mid + 1
            else:
                right_bound = mid

        indices.append(left)

    indices_tensor = torch.tensor(indices, dtype=torch.int32 if out_int32 else torch.int64)
    return indices_tensor


# 温度vs饱和压力转换
def tem_sat_press(press=None,tem=None):
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
    val_idx = searchsorted(tem_sat_press[:,mode], val) - 1
    # 确保索引在有效范围内
    val_idx = torch.clamp(val_idx, 0,  tem_sat_press.shape[0]- 2)

    def mode_reverse(mode):
        if mode == 0:
            return 1
        elif mode == 1:
            return 0
        else:
            print("mode error")
            return None

    output1 = tem_sat_press[val_idx, mode_reverse(mode)]

    output2 = tem_sat_press[val_idx+1, mode_reverse(mode)]

    val_w1 = tem_sat_press[val_idx, mode]
    val_w2 = tem_sat_press[val_idx+1, mode]


    w = (val - val_w1) / (val_w2 - val_w1)
    output = w * (output2 - output1) + output1
    return output

# 压缩机转速下限
def com_speed_min_bilinear_interpolation(lo_press, high_press):  # 双线性插值
    # 插值表
    lo_press_table = torch.tensor([100.0, 150, 200, 250, 300, 350, 400, 450, 500, 550])
    high_press_table = torch.tensor([200.0, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])

    # 第二个维度(高压压力)：200,400,600,800,1000,1200,1400,1600,1800,2000                     第一个维度(低压压力)
    com_speed_min_table = torch.tensor([[2000.0, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000],  # 100
                                        [1600, 1600, 1600, 1600, 1600, 1700, 1800, 1900, 2000, 2000],  # 150
                                        [1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1600, 2000],  # 200
                                        [900, 900, 950, 1000, 1050, 1100, 1150, 1200, 1600, 2000],  # 250
                                        [800, 800, 800, 800, 900, 1000, 1100, 1200, 1600, 2000],  # 300
                                        [800, 800, 800, 800, 800, 900, 1050, 1200, 1600, 2000],  # 350
                                        [800, 800, 800, 800, 800, 800, 1000, 1200, 1600, 2000],  # 400
                                        [800, 800, 800, 800, 800, 800, 950, 1200, 1600, 2000],  # 450
                                        [800, 800, 800, 800, 800, 800, 900, 1200, 1600, 2000],  # 500
                                        [800, 800, 800, 800, 800, 800, 850, 1200, 1600, 2000]])  # 550

    # 将输入的压力转换为 PyTorch 张量
    if isinstance(lo_press, list):
        lo_press = torch.tensor(lo_press)
    if isinstance(high_press, list):
        high_press = torch.tensor(high_press)
        # 确保输入张量是连续的
    lo_press = lo_press.contiguous()
    high_press = high_press.contiguous()

    # 找到输入低压和高压在表中的位置
    lo_press_idx = searchsorted(lo_press_table, lo_press) - 1
    high_press_idx = searchsorted(high_press_table, high_press) - 1

    # 确保索引在有效范围内
    lo_press_idx = torch.clamp(lo_press_idx, 0, len(lo_press_table) - 2)
    high_press_idx = torch.clamp(high_press_idx, 0, len(high_press_table) - 2)

    # 获取四个最近的点
    Q11 = com_speed_min_table[lo_press_idx, high_press_idx]
    Q12 = com_speed_min_table[lo_press_idx, high_press_idx + 1]
    Q21 = com_speed_min_table[lo_press_idx + 1, high_press_idx]
    Q22 = com_speed_min_table[lo_press_idx + 1, high_press_idx + 1]

    # 计算 x 和 y 方向的比例
    x_ratio = (lo_press - lo_press_table[lo_press_idx]) / (
            lo_press_table[lo_press_idx + 1] - lo_press_table[lo_press_idx])

    y_ratio = (high_press - high_press_table[high_press_idx]) / (
            high_press_table[high_press_idx + 1] - high_press_table[high_press_idx])

    # 在 x 方向上进行线性插值
    R1 = x_ratio * (Q21 - Q11) + Q11
    R2 = x_ratio * (Q22 - Q12) + Q12

    # 在 y 方向上进行线性插值
    P = y_ratio * (R2 - R1) + R1
    return P

def init_weights_xavier_uniform(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

class MyCabPosModel(nn.Module):
    def __init__(self, input_size=6,output_size=1, hidden_size1=None, hidden_size2=None, dropout = 0.2):
        super(MyCabPosModel,self).__init__()
        # 开度建模
        self.cab_pos_part1 = MLP(input_size=1, hidden_sizes=[512,512], output_size=1, use_batchnorm=True)
        self.cab_pos_part2 = MLP(input_size=1, hidden_sizes=[512, 512], output_size=1, use_batchnorm=True)
        self.cab_pos_part3 = MLP(input_size=5, hidden_sizes=[512,512], output_size=1, use_batchnorm=True)
        init_weights_xavier_uniform(self.cab_pos_part1)
        init_weights_xavier_uniform(self.cab_pos_part2)
        init_weights_xavier_uniform(self.cab_pos_part3)

    def forward(self, SCErr, exv_pid_iout, use_x):
        part1 = self.cab_pos_part1(SCErr.unsqueeze(1))
        part2 = self.cab_pos_part2(exv_pid_iout.unsqueeze(1))
        part3 = self.cab_pos_part3(use_x)
        cab_heating_status_act_pos = part1 + part2 + part3
        return cab_heating_status_act_pos[:,0]


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, use_batchnorm=True):
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

        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # 创建序列模型
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MyComSpeedModel(nn.Module):
    def __init__(self, input_size=6, output_size=1, hidden_size1=None, hidden_size2=None, hidden_size3=None, dropout = 0.5):
        super(MyComSpeedModel,self).__init__()
        # 压缩机建模
        # 不定系数n
        if hidden_size1 is None:
            hidden_size1 = [512, 512]
        if hidden_size2 is None:
            hidden_size2 = [512, 512]
        if hidden_size3 is None:
            hidden_size3 = [512, 512]

        self.com_speed_part1 = MLP(input_size=1, hidden_sizes=hidden_size1, output_size=1, use_batchnorm=True)

        self.com_speed_part2 = MLP(input_size=1, hidden_sizes=hidden_size2, output_size=1, use_batchnorm=True)

        self.com_speed_part3 = MLP(input_size=4, hidden_sizes=hidden_size3, output_size=1, use_batchnorm=True)

        init_weights_xavier_uniform(self.com_speed_part1)
        init_weights_xavier_uniform(self.com_speed_part2)
        init_weights_xavier_uniform(self.com_speed_part3)

    def forward(self, dif_hi_pressure, ac_pid_out_hp, use_x1):
        part1 = self.com_speed_part1(dif_hi_pressure.unsqueeze(1))
        part2 = self.com_speed_part2(ac_pid_out_hp.unsqueeze(1))
        part3 = self.com_speed_part3(use_x1)
        compressor_speed = part1 + part2 + part3
        return compressor_speed[:,0]





def norm(tensor,max,min):
    output = (tensor - min)/(max - min)
    return output

def renorm(tensor,max,min):
    output = tensor*(max - min) + min
    return output

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

        self.dif_hi_pressure_max = 2000.0
        self.dif_hi_pressure_min = -2000.0
        self.dif_lo_pressure_max = 2000.0
        self.dif_lo_pressure_min = -2000.0


        self.compressor_speed_model = MyComSpeedModel(input_size=6, output_size=1)
        self.cab_pos_model = MyCabPosModel(input_size=5,output_size=1)




    def forward(self, x):
        # 压缩机排气温度、内冷温度、饱和高压、压缩机进气温度、饱和低压、目标饱和高压、目标过冷度、目标过热度
        # 压缩机转速、膨胀阀开度
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        # 输出限制
        # 上一时刻ac_pid_pout
        ac_pid_out_hp = x[:,0]
        # 上一时刻饱和低压
        lo_pressure = x[:, 1]
        # 上一时刻饱和高压
        hi_pressure = x[:,2]
        # 目标饱和低压
        aim_lo_pressure = x[:,3]
        # 目标饱和高压
        aim_hi_pressure = x[:,4]
        # 上一时刻EXV_Oh_PID_Iout
        exv_pid_iout = x[:,5]
        # 目标过冷度
        sc_tar_mode_10 = x[:,6]
        # 目标过热度
        sh_tar_mode_10 = x[:,7]
        # 压缩机排气温度
        temp_p_h_2 = x[:,8]
        # 内冷温度
        temp_p_h_5 = x[:,9]
        # 压缩机进气温度
        temp_p_h_1_cab_heating = x[:,10]

        # 输入解耦
        # 压缩机温度差 = 压缩机排气温度 - 压缩机进气温度
        dif_temp_p_h = temp_p_h_2 - temp_p_h_1_cab_heating
        # 饱和高压差 = 目标饱和高压 - 饱和高压
        dif_hi_pressure = aim_hi_pressure - hi_pressure
        # 饱和低压差 = 目标饱和低压 - 饱和低压
        dif_lo_pressure = aim_lo_pressure - lo_pressure
        # 压缩比 = 饱和高压 / 饱和低压
        rate_pressure = hi_pressure / lo_pressure
        # 压缩差 = 饱和高压 - 饱和低压
        dif_pressure = hi_pressure - lo_pressure
        # # 压焓图1的焓值，排气温度焓值，制热
        # h_p_h_1_cab_heating = cal_h(temp_p_h_1_cab_heating.numpy(), lo_pressure.numpy(), states='gas')
        # # 压焓图2的焓值，进气温度焓值，制热
        # h_p_h_2_cab_heating = cal_h(temp_p_h_2.numpy(), hi_pressure.numpy(), states='gas')
        # 焓值差
        # h1_h2 = torch.clamp((torch.from_numpy(h_p_h_1_cab_heating) - torch.from_numpy(h_p_h_2_cab_heating)), min=1e-6)
        h1_h2 = torch.ones(temp_p_h_2.shape[0])
        # 实际过热度  = 压缩机排气温度 - 低压饱和压力对应温度
        sh_rel_mode_10 = temp_p_h_2 - tem_sat_press(lo_pressure)
        # 实际过冷度 =  高压饱和压力对应温度 - 内冷温度
        sc_rel_mode_10 = tem_sat_press(hi_pressure) - temp_p_h_5
        # SCErr 过热度偏差                             过冷度偏差
        SCErr = (sh_rel_mode_10 - sh_tar_mode_10) + (sc_rel_mode_10 - sc_tar_mode_10)


        # 计算最小理论压缩机转速
        com_speed_min = com_speed_min_bilinear_interpolation(lo_pressure, hi_pressure)

        """
            压缩机转速前向传播
        """
        use_x1 = torch.cat((hi_pressure.unsqueeze(1),lo_pressure.unsqueeze(1),aim_hi_pressure.unsqueeze(1),aim_lo_pressure.unsqueeze(1)), dim=1)

        compressor_speed = self.compressor_speed_model(dif_hi_pressure, ac_pid_out_hp, use_x1)

        """
            膨胀阀开度前向传播
        """
        # 排气 压缩机温度差 内冷 饱高 压缩比 包和高压差
        use_x = torch.cat((temp_p_h_2.unsqueeze(1), temp_p_h_5.unsqueeze(1), hi_pressure.unsqueeze(1),
                           temp_p_h_1_cab_heating.unsqueeze(1), lo_pressure.unsqueeze(1)), dim=1)
        cab_heating_status_act_pos = self.cab_pos_model(SCErr,exv_pid_iout,use_x)

        # 结果输出限幅
        if not self.training:
            compressor_speed = torch.round(compressor_speed).int()
            com_speed_min_mask = compressor_speed < com_speed_min
            compressor_speed[com_speed_min_mask] = com_speed_min[com_speed_min_mask].to(compressor_speed.dtype)

            cab_heating_status_act_pos = torch.round(cab_heating_status_act_pos).int()
        return torch.cat((compressor_speed.unsqueeze(1),cab_heating_status_act_pos.unsqueeze(1)),dim=1)



if __name__ == '__main__':
    # temp_p_h_1_cab_heating = torch.randn(1024,1).numpy()
    # lo_pressure = torch.randn(1024, 1).numpy()
    # temp_p_h_2 = torch.randn(1024, 1).numpy()
    # hi_pressure = torch.randn(1024, 1).numpy()
    # # 压焓图1的焓值，排气温度焓值，制热
    # h_p_h_1_cab_heating = cal_h(temp_p_h_1_cab_heating, lo_pressure, states='gas')
    # # 压焓图2的焓值，进气温度焓值，制热
    # h_p_h_2_cab_heating = cal_h(temp_p_h_2, hi_pressure, states='gas')
    # A = [375,225]
    # B = [1777,1234]
    # print(com_speed_min_bilinear_interpolation(A, B))

    A = [13.9,14.9,24.7]
    print(tem_sat_press(press=A))
# 1