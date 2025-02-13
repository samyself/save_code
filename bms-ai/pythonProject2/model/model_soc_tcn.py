import torch.nn as nn
import torch
from torch.nn import init
from ..common_func import init_weights_xavier_uniform, TemporalConvNet


class MyModel(nn.Module):
    def __init__(self, input_size=12, num_channels=[32, 16], output_size=1):
        super(MyModel, self).__init__()
        self.tcn1 = TemporalConvNet(input_size, num_channels, kernel_size=3)

        # 输出层
        self.fc = nn.Sequential(nn.Linear(num_channels[-1] * 60, 16),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.Linear(16, output_size),
                                )

        init_weights_xavier_uniform(self.tcn1)
        init_weights_xavier_uniform(self.fc)

        # 批归一化层
        self.bn = nn.BatchNorm1d(input_size)

    def forward(self, x):
        # x_in = ([Volt] + [Curr] + [Ah] + [P_vxi] + [avg_current60] + [avg_voltage60] +
        #         [avg_current300] + [avg_voltage300] + [avg_current600] + [avg_voltage600] +
        #         [avg_current_all] + [avg_voltage_all] + [BmsSoc])
        x = x[:, :, :-1]
        # 调整输入数据的形状
        batch_size, seq_len, input_size = x.size()
        x = x.view(batch_size * seq_len, input_size)  # (batch_size * seq_len, input_size)

        # 应用批归一化
        x = self.bn(x)

        # 恢复原始形状
        x = x.view(batch_size, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)

        # 传递给TCN
        out = self.tcn1(x)  # out: (batch_size,num_channels[-1], seq_lens)

        out = out.view(batch_size, -1)  # 将输出展平为 (batch_size, num_channels[-1]* seq_len)
        # 通过全连接层
        out = self.fc(out)  # out: (batch_size, 1)

        return out

