import torch.nn as nn
import torch
from torch.nn import init
from PycharmProjects.pythonProject2.common_func import init_weights_xavier_uniform, TemporalConvNet


class SocPreModel(nn.Module):
    def __init__(self, input_size=6, window_size=300, hidden_size1=64, hidden_size2=16, num_channels=[16],
                 num_layers1=1, num_layers2=1, output_size=2):
        super(SocPreModel, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers1 = num_layers1
        self.num_layers2 = num_layers2

        self.tcn1 = TemporalConvNet(input_size, num_channels, kernel_size=3)

        # 第一个 LSTM 层，return_sequences=True
        self.lstm1 = nn.LSTM(num_channels[-1], hidden_size1, num_layers1, batch_first=True)

        # 第二个 LSTM 层，return_sequences=False
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers2, batch_first=True)

        # 输出层
        self.fc = nn.Sequential(nn.Linear(hidden_size2 * 16, 64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Linear(64, 16),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.Linear(16, output_size)
                                )

        # 输出层
        self.l1 = nn.Sequential(nn.Linear(window_size, 16),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.Linear(16, output_size)
                                )
        self.l2 = nn.Sequential(nn.Linear(4, 64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Linear(64, 16),
                                nn.ReLU(),
                                nn.Linear(16, output_size)
                                )
        init_weights_xavier_uniform(self.lstm1)
        init_weights_xavier_uniform(self.lstm2)
        init_weights_xavier_uniform(self.fc)
        init_weights_xavier_uniform(self.l1)
        init_weights_xavier_uniform(self.l2)

        # 批归一化层
        self.bn = nn.BatchNorm1d(input_size)

    def forward(self, x1):
        # x_in =  ([Volt] + [Curr] + [Ah] + [P_vxi] + [DtAh] +
        #          [avg_current_all] + [avg_voltage_all] + [BmsSoc])
        Ah = x1[:, :, 2].reshape(x1.shape[0], -1)
        x = x1[:, :, [0, 1, 3, 4, 5, 6]]
        # 调整输入数据的形状
        batch_size, seq_len, input_size = x.size()
        x = x.view(batch_size * seq_len, input_size)  # (batch_size * seq_len, input_size)

        # 应用批归一化
        x = self.bn(x)

        # 恢复原始形状
        x = x.view(batch_size, seq_len, input_size)
        # 交换后两个维度
        x = x.transpose(1, 2)
        # 传递给TCN
        out1 = self.tcn1(x)  # out1: (batch_size,num_channels[-1], seq_lens)
        # 交换后两个维度
        out1 = out1.transpose(1, 2)  # out1: (batch_size, seq_lens，num_channels[-1])

        # 初始化第一个 LSTM 层的隐藏状态和细胞状态
        h0_1 = torch.zeros(self.num_layers1, out1.size(0), self.hidden_size1).to(out1.device)
        c0_1 = torch.zeros(self.num_layers1, out1.size(0), self.hidden_size1).to(out1.device)

        # 前向传播第一个 LSTM 层
        out2, _ = self.lstm1(out1, (h0_1, c0_1))  # out: (batch_size, seq_len, hidden_size1)

        # 初始化第二个 LSTM 层的隐藏状态和细胞状态
        h0_2 = torch.zeros(self.num_layers2, out1.size(0), self.hidden_size2).to(out1.device)
        c0_2 = torch.zeros(self.num_layers2, out1.size(0), self.hidden_size2).to(out1.device)

        # 前向传播第二个 LSTM 层
        out3, _ = self.lstm2(out2, (h0_2, c0_2))  # out: (batch_size, seq_len, hidden_size2)

        # 取最后一个时间步的输出
        out3 = out3[:, -16:, :].reshape(batch_size, -1)  # out: (batch_size, hidden_size2*16)

        part1 = self.fc(out3)
        part2 = self.l1(Ah)

        # 通过全连接层
        part1_2 = torch.cat((part1, part2), dim=1)
        out_all = self.l2(part1_2)

        return out_all


class SocDeltaModel(nn.Module):
    def __init__(self):
        super(SocDeltaModel, self).__init__()
        # 不确定度输出层
        self.l2 = nn.Sequential(nn.Linear(8, 16),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.Linear(16, 1),
                                # nn.Sigmoid()
                                )

    def forward(self, x):
        # x_in =  ([Volt] + [Curr] + [Ah] + [P_vxi] + [DtAh] +
        #          [avg_current_all] + [avg_voltage_all] + [BmsSoc])
        # x = x[:, :, [0, 1,2,3, 4, 5, 6]]
        out = self.l2(x)
        return out


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.SocPreModel = SocPreModel()
        self.SocDeltaModel = SocDeltaModel()

        # # 冻结 SocPreModel 的参数
        # for param in self.SocPreModel.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = x.clone().detach()
        out1 = self.SocPreModel(x1)[:, 0].reshape(batch_size, -1)
        out1 = torch.clamp(out1, min=0.0, max=99.99)
        out2 = self.SocDeltaModel(x[:, -1, :].reshape(batch_size, -1))
        out2 = torch.clamp(out2, min=-10.0, max=10.0)
        out = torch.cat((out1, out2), dim=1)
        return out

if __name__ == '__main__':
    net = MyModel()
    soc_pre_model_path = '../../data/ckpt/Soc_0102_lstm_tcn_v1_3_delta.pth'
    net.load_state_dict(torch.load(soc_pre_model_path, map_location=torch.device('cpu')))
    net.eval()
    input = torch.randn(1,300,8)
    y = net(input)
    print(y)