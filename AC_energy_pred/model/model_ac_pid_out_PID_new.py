import torch
import torch.nn as nn
import torch.nn.init as init

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
        values = values[:,0]

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
# 单线性插值
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

# 双线性插值
def inter2D(x1_table,x2_table, y_table, x1, x2):  # 双线性插值
    # 将输入的压力转换为 PyTorch 张量
    if isinstance(x1, list):
        x1 = torch.tensor(x1)
    if isinstance(x2, list):
        x2 = torch.tensor(x2)
        # 确保输入张量是连续的
    x1 = x1.contiguous()
    x2 = x2.contiguous()

    # 找到输入低压和高压在表中的位置
    x1_idx = searchsorted(x1_table, x1) - 1
    x2_idx = searchsorted(x2_table, x2) - 1

    # 确保索引在有效范围内
    x1_idx = torch.clamp(x1_idx, 0, len(x1_table) - 2)
    x2_idx = torch.clamp(x2_idx, 0, len(x2_table) - 2)

    # 获取四个最近的点
    Q11 = y_table[x1_idx, x2_idx]
    Q12 = y_table[x1_idx, x2_idx + 1]
    Q21 = y_table[x1_idx + 1, x2_idx]
    Q22 = y_table[x1_idx + 1, x2_idx + 1]

    # 计算 x 和 y 方向的比例
    x_ratio = (x1 - x1_table[x1_idx]) / (
            x1_table[x1_idx + 1] - x1_table[x1_idx])

    y_ratio = (x2 - x2_table[x2_idx]) / (
            x2_table[x2_idx + 1] - x2_table[x2_idx])

    # 在 x 方向上进行线性插值
    R1 = x_ratio * (Q21 - Q11) + Q11
    R2 = x_ratio * (Q22 - Q12) + Q12

    # 在 y 方向上进行线性插值
    P = y_ratio * (R2 - R1) + R1
    return P



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
        self.PID_Kp_list = torch.tensor([0.5, 0.3417968, 0.3417968, 0.3417968, 0.3417968,0.5], device=device)
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


        self.AC_Kp_Rate =  nn.Parameter(torch.rand(1, device=device), requires_grad=True)
        self.Kd_paras = torch.tensor([0.048828], device=device)

        self.Ivalue = torch.tensor([0.0], device=device)
        self.last_Pout = torch.tensor([0.0], device=device)
        self.last_Dout = torch.tensor([0.0], device=device)

        self.last_diff_hp = 0

        self.MLP1 = MLP(input_size=4, hidden_sizes=[512, 512], output_size=1,use_batchnorm=False,use_dropout=False)
        # self.MLP2 = MLP(input_size=11, hidden_sizes=[256, 256], output_size=1,use_batchnorm=False,use_dropout=True)

        init_weights_xavier_uniform(self.MLP1)
        # init_weights_xavier_uniform(self.MLP2)

        self.Iout_bais = MLP(input_size=4, hidden_sizes=[512, 512], output_size=1,use_batchnorm=False,use_dropout=False)
        self.Diffout_bais = MLP(input_size=4, hidden_sizes=[512, 512], output_size=1, use_batchnorm=False,
                             use_dropout=False)
        init_weights_xavier_uniform(self.Iout_bais)
        init_weights_xavier_uniform(self.Diffout_bais)

        self.MLP3 = MLP(input_size=6, hidden_sizes=[256, 256], output_size=1, use_batchnorm=False, use_dropout=False)
        self.MLP4 = MLP(input_size=2, hidden_sizes=[256, 256], output_size=1, use_batchnorm=False, use_dropout=False)

        init_weights_xavier_uniform(self.MLP3)
        init_weights_xavier_uniform(self.MLP4)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        last_ac_pid_out_hp = x[:,0]
        # 上一时刻饱和低压
        lo_pressure = x[:, 1]
        # 上一时刻饱和高压
        hi_pressure = x[:, 2]
        # 目标饱和高压
        aim_hi_pressure = x[:, 3]

        # 压缩机排气温度
        temp_p_h_2 = x[:, 4]
        # 内冷温度
        temp_p_h_1_cab_heating = x[:, 5]
        # 压缩机进气温度
        temp_p_h_5 = x[:, 6]



       #高压偏差
        diff_hi_pressure = (aim_hi_pressure-hi_pressure)
        # 压缩机温度差 = 压缩机排气温度 - 压缩机进气温度
        dif_temp_p_h = temp_p_h_2 - temp_p_h_1_cab_heating


        com_speed_min = inter2D(self.lo_press_table, self.high_press_table, self.com_speed_min_table, lo_pressure, hi_pressure)


        use_x1 = torch.cat((temp_p_h_2.unsqueeze(1), temp_p_h_5.unsqueeze(1), hi_pressure.unsqueeze(1),
                            temp_p_h_1_cab_heating.unsqueeze(1), lo_pressure.unsqueeze(1),
                            aim_hi_pressure.unsqueeze(1)), dim=1)

        Part1 = self.MLP3(use_x1)

        use_x2 = torch.cat((dif_temp_p_h.unsqueeze(1), diff_hi_pressure.unsqueeze(1)), dim=1)

        Part2 = self.MLP4(use_x2)

        ac_pid_out_hp = last_ac_pid_out_hp + Part1 + Part2
        # 限定最小值为com_speed_min
        ac_pid_out_hp = torch.relu(ac_pid_out_hp - com_speed_min) + com_speed_min
        # 限定最大值为8000
        ac_pid_out_hp = 8000 - torch.relu(8000 - ac_pid_out_hp)

        return ac_pid_out_hp

if __name__ == '__main__':

    # ac_pid_out_hp = torch.tensor([7999,8001,8000])
    # com_speed_min = torch.tensor([8000,8000,8000])
    # # ac_pid_out_hp = torch.relu(ac_pid_out_hp - com_speed_min) + com_speed_min
    # # ac_pid_out_hp = 8000 - torch.relu(8000 - ac_pid_out_hp)
    #
    # print(ac_pid_out_hp)

    diffpress = [-11,-12,11,12]
    PID_Kp_list = torch.tensor([0.5, 0.3417968, 0.3417968, 0.3417968, 0.3417968, 0.5])
    diffpress_Kp_list = torch.tensor([-10, -2.5, -1, 1, 2.5, 10])
    print(inter1D(diffpress_Kp_list, PID_Kp_list,diffpress))

