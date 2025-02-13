import torch
import torch.nn as nn
import torch.nn.init as init


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

# 双线性插值
def inter2D(x1_table,x2_table, y_table, x1, x2):  # 双线性插值
    # 插值表
    # lo_press_table = torch.tensor([100.0, 150, 200, 250, 300, 350, 400, 450, 500, 550])
    # high_press_table = torch.tensor([200.0, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    #
    # # 第二个维度(高压压力)：200,400,600,800,1000,1200,1400,1600,1800,2000                     第一个维度(低压压力)
    # com_speed_min_table = torch.tensor([[2000.0, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000],  # 100
    #                                     [1600, 1600, 1600, 1600, 1600, 1700, 1800, 1900, 2000, 2000],  # 150
    #                                     [1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1600, 2000],  # 200
    #                                     [900, 900, 950, 1000, 1050, 1100, 1150, 1200, 1600, 2000],  # 250
    #                                     [800, 800, 800, 800, 900, 1000, 1100, 1200, 1600, 2000],  # 300
    #                                     [800, 800, 800, 800, 800, 900, 1050, 1200, 1600, 2000],  # 350
    #                                     [800, 800, 800, 800, 800, 800, 1000, 1200, 1600, 2000],  # 400
    #                                     [800, 800, 800, 800, 800, 800, 950, 1200, 1600, 2000],  # 450
    #                                     [800, 800, 800, 800, 800, 800, 900, 1200, 1600, 2000],  # 500
    #                                     [800, 800, 800, 800, 800, 800, 850, 1200, 1600, 2000]])  # 550

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

# 单线性插值
def inter1D(x_list, y_list, x):
    if not isinstance(x_list, torch.Tensor):
        x_list = torch.tensor([x_list])
    if not isinstance(y_list, torch.Tensor):
        y_list = torch.tensor([y_list])
    if not isinstance(x, torch.Tensor):
        x = torch.tensor([x])

        # 确保输入张量是连续的
    x = x.contiguous()

    # 找到输入低压和高压在表中的位置
    x_index = searchsorted(x_list, x) - 1

    y = y_list[x_index] + (x - x_list[x_index]) * (y_list[x_index + 1] - y_list[x_index]) / (x_list[x_index + 1] - x_list[x_index])
    return y

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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class UniTSModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dropout=0.1):
        super(UniTSModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout),
            num_layers
        )
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        x = self.output_proj(x)
        return x


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
                                        [32, 32, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,]])            # 32

        self.MLP1 = MLP(input_size=4, hidden_sizes=[1024, 1024], output_size=1,use_batchnorm=True,use_dropout=True)
        self.MLP2 = MLP(input_size=6, hidden_sizes=[1024, 1024], output_size=1,use_batchnorm=True,use_dropout=True)
        self.MLP3 = MLP(input_size=8, hidden_sizes=[1024, 1024], output_size=1, use_batchnorm=True, use_dropout=True)


        init_weights_xavier_uniform(self.MLP1)
        init_weights_xavier_uniform(self.MLP2)
        init_weights_xavier_uniform(self.MLP3)

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
        # 当前后排设定温度
        cab_rl_set_temp = x[:,4]

        # 当前主驾驶目标进风温度
        cab_fl_req_temp = x[:,5]
        # 当前副驾驶目标进风温度
        cab_fr_req_temp = x[:,6]
        # 当前后排目标进风温度
        cab_rl_req_temp = x[:,7]

        # 上一时刻饱和低压
        lo_pressure = x[:,8]
        # 上一时刻饱和高压
        hi_pressure = x[:,9]
        # 目标饱和低压
        aim_lo_pressure = x[:,10]
        # 目标饱和高压
        aim_hi_pressure = x[:,11]

        # 当前乘务舱温度
        temp_incar = x[:,12]
        # 电池请求冷却液温度
        temp_battery_req = x[:,13]
        # 冷却液进入电池的温度
        temp_coolant_battery_in = x[:,14]

        # 冷却液进入电池的温度
        ac_kp_rate_last = x[:,15]

        # 高压偏差
        diff_hi_pressure = (hi_pressure - aim_hi_pressure)

        #   DVT_CabinErr_FL = f(主驾设定温度, 环境温度) 查表 - SEN_Incar(乘员舱温度)
        DVT_CabinErr_FL = inter2D(self.temp_set,self.temp_envr,self.CabinSP_table,cab_fl_set_temp,temp_amb) - temp_incar
        #   DVT_CabinErr_FR = f(副驾设定温度, 环境温度) 查表 - SEN_Incar(乘员舱温度)
        DVT_CabinErr_FR = inter2D(self.temp_set, self.temp_envr, self.CabinSP_table,cab_fr_set_temp, temp_amb) - temp_incar
        # DCT_DCT_ChillerTempErr = HP_Mode_BattWTReqOpt（电池请求冷却的温度） - SEN_CoolT_Battln（冷却液进入电池的温度）
        DCT_ChillerTempErr  = temp_battery_req - temp_coolant_battery_in




        # norm
        # last_ac_pid_out_hp = norm(last_ac_pid_out_hp, self.ac_pid_out_hp_max, self.ac_pid_out_hp_min)
        # temp_amb = norm(temp_amb, self.temp_amb_max, self.temp_amb_min)
        # cab_fl_set_temp = norm(cab_fl_set_temp, self.cab_set_temp_max, self.cab_set_temp_min)
        # cab_fr_set_temp = norm(cab_fr_set_temp, self.cab_set_temp_max, self.cab_set_temp_min)
        # cab_rl_set_temp = norm(cab_rl_set_temp, self.cab_set_temp_max, self.cab_set_temp_min)
        # cab_fl_req_temp = norm(cab_fl_req_temp, self.cab_req_temp_max, self.cab_req_temp_min)
        # cab_fr_req_temp = norm(cab_fr_req_temp, self.cab_req_temp_max, self.cab_req_temp_min)
        # cab_rl_req_temp = norm(cab_rl_req_temp, self.cab_req_temp_max, self.cab_req_temp_min)
        # lo_pressure = norm(lo_pressure, self.lo_pressure_max, self.lo_pressure_min)
        # hi_pressure = norm(hi_pressure, self.hi_pressure_max, self.hi_pressure_min)
        # aim_lo_pressure = norm(aim_lo_pressure, self.lo_pressure_max, self.lo_pressure_min)
        # aim_hi_pressure = norm(aim_hi_pressure, self.hi_pressure_max, self.hi_pressure_min)

        # use_x1 = torch.cat((DVT_CabinErr_FL.unsqueeze(1), DVT_CabinErr_FR.unsqueeze(1), temp_amb.unsqueeze(1), DCT_ChillerTempErr .unsqueeze(1)), dim=1)

        # AC_KpRate = self.MLP1(use_x1)

        # use_x2 = torch.cat((ac_kp_rate_last, diff_hi_pressure.unsqueeze(1),
        #                     lo_pressure.unsqueeze(1), hi_pressure.unsqueeze(1),aim_lo_pressure.unsqueeze(1),aim_hi_pressure.unsqueeze(1)), dim=1)
        # pid_paras = self.MLP2(use_x2)

        Kp = inter1D(self.diffpress_Kp_list, self.PID_Kp_list, diff_hi_pressure/1000)*ac_kp_rate_last
        Ki = inter1D(self.diffpress_Ki_list, self.PID_Ki_list, diff_hi_pressure/1000)*ac_kp_rate_last
        Kd = self.Kd_paras*ac_kp_rate_last


        use_x3 = torch.cat((Kp.unsqueeze(1),Ki.unsqueeze(1),Kd.unsqueeze(1), diff_hi_pressure.unsqueeze(1),
                            lo_pressure.unsqueeze(1), hi_pressure.unsqueeze(1),aim_lo_pressure.unsqueeze(1),aim_hi_pressure.unsqueeze(1)), dim=1)
        ac_pid_out_hp = self.MLP3(use_x3)

        # if not self.training:
        #     compressor_speed = torch.round(compressor_speed).int()
        #     com_speed_min_mask = compressor_speed < com_speed_min
        #     compressor_speed[com_speed_min_mask] = com_speed_min[com_speed_min_mask].to(compressor_speed.dtype)
        #
        #     cab_heating_status_act_pos = torch.round(cab_heating_status_act_pos).int()
        return ac_pid_out_hp[:,0]