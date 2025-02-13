import torch
import torch.nn as nn
import torch.nn.init as init
import math

#梯度裁剪
# def clip_wights(self):
#     for layer in self.network:
#         if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
#             with torch.no_grad():
#                 layer.weight.clamp_(-65536, 65536)
#                 if layer.bias is not None:
#                     layer.bias.clamp_(-65536, 65536)


# 相关度torch版本，nan会输出0
def XiangGuanDu(output, target):
    mean1 = torch.mean(output)
    mean2 = torch.mean(target)

    # 去均值
    tensor1_centered = output - mean1
    tensor2_centered = target - mean2

    # 计算协方差
    covariance = torch.dot(tensor1_centered, tensor2_centered) / (output.size(0) - 1)

    # 计算标准差
    std1 = torch.std(output, unbiased=True)
    std2 = torch.std(target, unbiased=True)

    # 计算皮尔逊相关系数
    pearson_correlation = covariance / (std1 * std2 + 1e-4)

    return pearson_correlation

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
    val_idx = torch.searchsorted(tem_sat_press[:, mode].contiguous(), val)
    # 确保索引在有效范围内
    val_idx = torch.clamp(val_idx, 0, tem_sat_press.shape[0] - 2)

    def mode_reverse(mode):
        if mode == 0:
            return 1
        elif mode == 1:
            return 0
        else:
            print("mode error")
            return None

    output1 = tem_sat_press[val_idx, mode_reverse(mode)]

    output2 = tem_sat_press[val_idx + 1, mode_reverse(mode)]

    val_w1 = tem_sat_press[val_idx, mode]
    val_w2 = tem_sat_press[val_idx + 1, mode]

    w = (val - val_w1) / (val_w2 - val_w1)
    output = w * (output2 - output1) + output1
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        self.d_v = d_model // num_heads  # 每个头的维度

        # 定义线性变换层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)  # 最终的线性变换层

    def forward(self, X, mask=None):
        """
        Args:
            X: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 掩码张量，用于掩盖某些位置，形状为 (batch_size, 1, 1, seq_len)

        Returns:
            output: 输出张量，形状为 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = X.size()

        # 线性变换
        Q = self.W_Q(X)  # (batch_size, seq_len, d_model)
        K = self.W_K(X)  # (batch_size, seq_len, d_model)
        V = self.W_V(X)  # (batch_size, seq_len, d_model)

        # 将 Q, K, V 分割成多个头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,
                                                                            2)  # (batch_size, num_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,
                                                                            2)  # (batch_size, num_heads, seq_len, d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1,
                                                                            2)  # (batch_size, num_heads, seq_len, d_v)

        # 计算相似度
        S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch_size, num_heads, seq_len, seq_len)

        # 应用掩码（如果有）
        if mask is not None:
            S = S.masked_fill(mask == 0, -1e9)  # 掩码的位置设为负无穷大

        # 归一化
        A = torch.softmax(S, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

        # 加权求和
        O = torch.matmul(A, V)  # (batch_size, num_heads, seq_len, d_v)

        # 拼接多个头的输出
        O = O.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)

        # 最终的线性变换
        output = self.W_O(O)  # (batch_size, seq_len, d_model)

        return output

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

    # 直接将表格插值补齐
    x1_min = torch.min(x1)
    x1_max = torch.max(x1)
    #y_table == x1_len * x2_len
    # 在x1左边添加一行，在y上面添加一行
    if x1_min<x1_table[0]:
        y_left = (y_table[1,:] - y_table[0,:])/(x1_table[1] - x1_table[0]) * (x1_min - x1_table[0]) + y_table[0,:]
        y_table[0,:] = y_left
        x1_table[0] = x1_min
    # 在x1右边添加一行，在y下面添加一行
    if x1_max>x1_table[-1]:
        y_right = (y_table[-1,:] - y_table[-2,:])/(x1_table[-1] - x1_table[-2]) * (x1_max - x1_table[-1]) + y_table[-1,:]
        y_table[-1,:] = y_right
        x1_table[-1] = x1_max

    x2_min = torch.min(x2)
    x2_max = torch.max(x2)
    #在x2左边添加一行，在y左边添加一行
    if x2_min<x2_table[0]:
        y_left = (y_table[:,1] - y_table[:,0])/(x2_table[1] - x2_table[0]) * (x2_min - x2_table[0]) + y_table[:,0]
        y_table[:,0] = y_left
        x2_table[0] = x2_min
    #在x2左边添加一行，在y右边添加一行
    if x2_max>x2_table[-1]:
        y_left = (y_table[:,-1] - y_table[:,-2])/(x2_table[-1] - x2_table[-2]) * (x2_max - x2_table[-1]) + y_table[:,-1]
        y_table[:,-1] = y_left
        x2_table[-1] = x2_max

    # 创建一个网格，用于查找

    # 找到输入低压和高压在表中的位置
    x1_idx = searchsorted(x1_table, x1)
    x2_idx = searchsorted(x2_table, x2)

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


# 双线性插值
def inter2D2(x1_table, x2_table, y_table, x1, x2):  # 双线性插值

    # 将输入的压力转换为 PyTorch 张量
    if isinstance(x1, list):
        x1 = torch.tensor(x1)
    if isinstance(x2, list):
        x2 = torch.tensor(x2)
        # 确保输入张量是连续的
    x1 = x1.contiguous()
    x2 = x2.contiguous()

    # 初始化输出张量
    y_inter = torch.zeros_like(x1, dtype=torch.float32)
    # 找到输入低压和高压在表中的位置
    x1_idx = searchsorted(x1_table, x1)
    x2_idx = searchsorted(x2_table, x2)

    # 处理边界条件
    x1_outside_left = x1_idx < 0
    x1_outside_right = x1_idx >= len(x1_table) - 1
    x2_outside_left = x2_idx < 0
    x2_outside_right = x2_idx >= len(x2_table) - 1

    # 对于在表格范围内的值，进行双线性插值
    mask_in_range = ~(x1_outside_left | x1_outside_right | x2_outside_left | x2_outside_right)
    x1_idx_in_range = x1_idx[mask_in_range]
    x2_idx_in_range = x2_idx[mask_in_range]
    x1_in_range = x1[mask_in_range]
    x2_in_range = x2[mask_in_range]

    y00 = y_table[x1_idx_in_range, x2_idx_in_range]
    y01 = y_table[x1_idx_in_range, x2_idx_in_range + 1]
    y10 = y_table[x1_idx_in_range + 1, x2_idx_in_range]
    y11 = y_table[x1_idx_in_range + 1, x2_idx_in_range + 1]

    w_x1 = (x1_in_range - x1_table[x1_idx_in_range]) / (x1_table[x1_idx_in_range + 1] - x1_table[x1_idx_in_range])
    w_x2 = (x2_in_range - x2_table[x2_idx_in_range]) / (x2_table[x2_idx_in_range + 1] - x2_table[x2_idx_in_range])

    y_inter[mask_in_range] = (1 - w_x1) * ((1 - w_x2) * y00 + w_x2 * y01) + w_x1 * ((1 - w_x2) * y10 + w_x2 * y11)

    # 对于在表格范围外的值，进行线性外推
    if x1_outside_left.any():
        y_inter[x1_outside_left] = y_table[0, x2_idx[x1_outside_left]] + (x1[x1_outside_left] - x1_table[0]) * (
                    y_table[1, x2_idx[x1_outside_left]] - y_table[0, x2_idx[x1_outside_left]]) / (
                                               x1_table[1] - x1_table[0])

    if x1_outside_right.any():
        y_inter[x1_outside_right] = y_table[-2, x2_idx[x1_outside_right]] + (x1[x1_outside_right] - x1_table[-2]) * (
                    y_table[-1, x2_idx[x1_outside_right]] - y_table[-2, x2_idx[x1_outside_right]]) / (
                                                x1_table[-1] - x1_table[-2])

    if x2_outside_left.any():
        y_inter[x2_outside_left] = y_table[x1_idx[x2_outside_left], 0] + (x2[x2_outside_left] - x2_table[0]) * (
                    y_table[x1_idx[x2_outside_left], 1] - y_table[x1_idx[x2_outside_left], 0]) / (
                                               x2_table[1] - x2_table[0])

    if x2_outside_right.any():
        y_inter[x2_outside_right] = y_table[x1_idx[x2_outside_right], -2] + (x2[x2_outside_right] - x2_table[-2]) * (
                    y_table[x1_idx[x2_outside_right], -1] - y_table[x1_idx[x2_outside_right], -2]) / (
                                                x2_table[-1] - x2_table[-2])

    return y_inter


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
    x_index = searchsorted(x_list, x)

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

def init_weights_xavier_uniform(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

def norm(tensor, max, min):
    output = (tensor - min) / (max - min)
    return output


def renorm(tensor, max, min):
    output = tensor * (max - min) + min
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
        return self.network(x)

class LSTM(nn.Module):
    def __init__(self, input_dim,output_dim, hidden_size=128, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        # 定义lsmt层
        # batch_first=True表示输入数据的形状是(batch_size, sequence_length, input_size)
        # 而不是默认的(sequence_length, batch_size, input_size)。
        # batch_size是指每个训练批次中包含的样本数量
        # sequence_length是指输入序列的长度
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)

        self.mlp = MLP(input_size=hidden_size, hidden_sizes=[1024,1024], output_size=output_dim, use_batchnorm=True, use_dropout=True)
        self.mlp.apply(init_weights_xavier_uniform)

        # 初始化LSTM的权重
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)  # 使用Xavier均匀分布初始化输入权重
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)  # 使用正交矩阵初始化隐藏层权重
            elif 'bias' in name:
                init.constant_(param.data, 0)  # 初始化偏置为0

        # 初始化全连接层的权重
        # init.kaiming_uniform_(self.fc.weight.data)  # 使用He均匀分布初始化全连接层权重
        # init.constant_(self.fc.bias.data, 0)  # 初始化全连接层偏置为0

    def forward(self, x):
        batch_size = x.shape[0]
        # 创建LSTM网络
        # 初始化了隐藏状态h0和细胞状态c0，并将其设为零向量。
        h0 = torch.zeros(self.num_layers,batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers,batch_size, self.hidden_size).to(x.device)

        # LSTM层前向传播
        # 将输入数据x以及初始化的隐藏状态和细胞状态传入LSTM层
        # 得到输出out和更新后的状态。
        # out的形状为(batch_size, sequence_length, hidden_size)。
        out, _ = self.lstm(x, (h0, c0))
        # 全连接层前向传播
        out = self.mlp(out[:,-1, :])
        return out

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.5):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        y = self.relu(out + res)
        # 检查中间结果是否包含 NaN
        if torch.isnan(y).any():
            print("NaN detected in TemporalBlock output.")
        return y


# TCN网络
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.5):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        y = self.network(x)
        # 在 TCN 输出后也检查是否包含 NaN
        assert not torch.isnan(y).any(), "NaN detected in TCN output."
        return y

if __name__ == '__main__':
    X1_list = torch.tensor([1 , 2 , 3 , 4 , 5 ])
    X2_list = torch.tensor([17,18,19])
    Y_list = torch.tensor([[ 1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])

    x1 = [0, 1,5,6]
    x2 = [20,19,17,16]

    print(inter2D(X1_list,X2_list,Y_list,x1,x2))
    print(inter2D2(X1_list, X2_list, Y_list, x1, x2))