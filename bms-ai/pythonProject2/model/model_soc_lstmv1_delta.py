import torch.nn as nn
import torch
from torch.nn import init
from PycharmProjects.pythonProject2.common_func import init_weights_xavier_uniform


class SocPreModel(nn.Module):
    def __init__(self, input_size=6, hidden_size1=256, hidden_size2=64, num_layers1=1, num_layers2=1, output_size=3):
        super(SocPreModel, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers1 = num_layers1
        self.num_layers2 = num_layers2

        # 第一个 LSTM 层，return_sequences=True
        self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers1, batch_first=True)

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
        self.l1 = nn.Sequential(nn.Linear(1, 16),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.Linear(16, output_size)
                                )

        init_weights_xavier_uniform(self.lstm1)
        init_weights_xavier_uniform(self.lstm2)
        init_weights_xavier_uniform(self.fc)
        init_weights_xavier_uniform(self.l1)

        # 批归一化层
        self.bn = nn.BatchNorm1d(input_size)

    def forward(self, x):
        with torch.no_grad():
            # x_in =  ([Volt] + [Curr] + [Ah] + [P_vxi] + [DtAh] +
            #          [avg_current_all] + [avg_voltage_all] + [BmsSoc])
            Ah = x[:, -1, 2].view(-1, 1)
            x = x[:, :, [0, 1, 3, 4, 5, 6]]
            # 调整输入数据的形状
            batch_size, seq_len, input_size = x.size()
            x = x.view(batch_size * seq_len, input_size)  # (batch_size * seq_len, input_size)

            # 应用批归一化
            x = self.bn(x)

            # 恢复原始形状
            x = x.view(batch_size, seq_len, input_size)

            # 初始化第一个 LSTM 层的隐藏状态和细胞状态
            h0_1 = torch.zeros(self.num_layers1, x.size(0), self.hidden_size1).to(x.device)
            c0_1 = torch.zeros(self.num_layers1, x.size(0), self.hidden_size1).to(x.device)

            # 前向传播第一个 LSTM 层
            out, _ = self.lstm1(x, (h0_1, c0_1))  # out: (batch_size, seq_len, hidden_size1)

            # 初始化第二个 LSTM 层的隐藏状态和细胞状态
            h0_2 = torch.zeros(self.num_layers2, x.size(0), self.hidden_size2).to(x.device)
            c0_2 = torch.zeros(self.num_layers2, x.size(0), self.hidden_size2).to(x.device)

            # 前向传播第二个 LSTM 层
            out, _ = self.lstm2(out, (h0_2, c0_2))  # out: (batch_size, seq_len, hidden_size2)

            # 取最后一个时间步的输出
            out = out[:, -16:, :].reshape(batch_size, -1)  # out: (batch_size, hidden_size2*16)

            # 通过全连接层
            out = self.fc(out) + self.l1(Ah)  # out: (batch_size, 3)
        return out


class SocDeltaModel(nn.Module):
    def __init__(self):
        super(SocDeltaModel, self).__init__()
        # 不确定度输出层
        self.l2 = nn.Sequential(nn.Linear(8, 16),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.Linear(16, 1)
                                )

    def forward(self, x):
        out = self.l2(x)
        return out


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.SocPreModel = SocPreModel()
        self.SocDeltaModel = SocDeltaModel()

        # 冻结 SocPreModel 的参数
        for param in self.SocPreModel.parameters():
            param.requires_grad = False

    def forward(self, x):
        batch_size = x.shape[0]
        out1 = self.SocPreModel(x)[:,0].reshape(batch_size, -1)
        out1 = out1.detach()
        out1 = torch.clamp(out1, min=0.0, max=99.99)

        out2 = self.SocDeltaModel(x[:, -1, :].reshape(batch_size, -1))
        out2 = torch.clamp(out2, min=-10.0, max=10.0)
        out = torch.cat((out1, out2), dim=1)
        return out

# class

if __name__ == '__main__':
    net = MyModel()
    soc_pre_model_path = '../../data/ckpt/Soc_0102_lstm_v1_2_delta.pth'
    net.load_state_dict(torch.load(soc_pre_model_path, map_location=torch.device('cpu')))
    net.eval()
    input = torch.randn(1,300,8)
    y = net(input)
    print(y)
