import torch
import torch.nn as nn
from numpy.array_api import zeros
from torch.nn.utils import weight_norm
from torch.onnx.symbolic_opset9 import tensor


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

def norm(tensor,max,min):
    output = (tensor - min)/(max - min)
    return output

class tcn_net(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(tcn_net, self).__init__()

        self.tcn1 = TemporalConvNet(input_size, num_channels, kernel_size = kernel_size, dropout = dropout)
        self.linear1 = nn.Sequential(nn.Linear(num_channels[-1], 1024),nn.ReLU(),nn.Dropout(0.5),
                                     nn.Linear(1024, 1024),nn.ReLU(),nn.Dropout(0.5),
                                     nn.Linear(1024, 1)
                                     )
        self.tcn2 = TemporalConvNet(input_size, num_channels, kernel_size = kernel_size, dropout = dropout)
        self.linear2 = nn.Sequential(nn.Linear(num_channels[-1], 1024),nn.ReLU(),nn.Dropout(0.5),
                                     nn.Linear(1024, 1024),nn.ReLU(),nn.Dropout(0.5),
                                     nn.Linear(1024, 1)
                                     )
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

        self.last1_usex = torch.zeros([1,input_size])
        self.last2_usex = torch.zeros([1,input_size])
        self.last3_usex = torch.zeros([1,input_size])
        self.last4_usex = torch.zeros([1,input_size])
        # self.last_data = {}
        # for i in range(1,kernel_size):
        #     for j in range(1,output_size+1):
        #         self.last_data[f'last{i}_usex{j}'] = torch.zeros([1,6])


    def forward(self, x):
        # 输出限制
        # 压缩机排气温度
        temp_p_h_2 = x[:, 0]
        # temp_p_h_2 = torch.clamp(x[:,0],max=self.temp_p_h_2_max,min=self.temp_p_h_2_min)
        temp_p_h_2 = norm(temp_p_h_2, max=self.temp_p_h_2_max, min=self.temp_p_h_2_min)
        # 内冷温度
        temp_p_h_5 = x[:, 1]
        # temp_p_h_5 = torch.clamp(x[:,1],max=self.temp_p_h_5_max,min=self.temp_p_h_5_min)
        temp_p_h_5 = norm(temp_p_h_5, max=self.temp_p_h_5_max, min=self.temp_p_h_5_min)
        # 饱和高压
        hi_pressure = x[:, 2]
        hi_pressure_dao = 1/hi_pressure
        # hi_pressure = torch.clamp(x[:,2],max=self.hi_pressure_max,min=self.hi_pressure_min)
        hi_pressure = norm(hi_pressure, max=self.hi_pressure_max, min=self.hi_pressure_min)
        # 压缩机进气温度
        temp_p_h_1_cab_heating = x[:, 3]
        # temp_p_h_1_cab_heating = torch.clamp(x[:,3],max=self.temp_p_h_1_cab_heating_max,min=self.temp_p_h_1_cab_heating_min)
        temp_p_h_1_cab_heating = norm(temp_p_h_1_cab_heating, max=self.temp_p_h_1_cab_heating_max,
                                      min=self.temp_p_h_1_cab_heating_min)
        # 饱和低压
        lo_pressure = x[:, 4]
        # lo_pressure = torch.clamp(x[:,4],max=self.lo_pressure_max,min=self.lo_pressure_min)
        lo_pressure = norm(lo_pressure, max=self.lo_pressure_max, min=self.lo_pressure_min)
        # 目标饱和高压
        aim_hi_pressure = x[:, 5]
        # aim_hi_pressure = torch.clamp(x[:,5],max=self.aim_hi_pressure_max,min=self.aim_hi_pressure_min)
        aim_hi_pressure = norm(aim_hi_pressure, max=self.aim_hi_pressure_max, min=self.aim_hi_pressure_min)

        # 输入解耦
        # 压缩机温度差 = 压缩机排气温度 - 压缩机进气温度
        dif_temp_p_h = temp_p_h_2 - temp_p_h_1_cab_heating
        # 饱和高压差 = 目标饱和高压 - 饱和高压
        dif_hi_pressure = aim_hi_pressure - hi_pressure
        # 压缩比 = 饱和高压 / 饱和低压
        rate_pressure = hi_pressure / (lo_pressure+1e-6)

        # 进气 压缩机温度差 内冷 饱高 压缩比 包和高压差
        use_x = torch.cat((temp_p_h_2.unsqueeze(1),dif_temp_p_h.unsqueeze(1),temp_p_h_5.unsqueeze(1),
                            hi_pressure.unsqueeze(1),rate_pressure.unsqueeze(1),dif_hi_pressure.unsqueeze(1),hi_pressure_dao.unsqueeze(1)), dim=1)
        if use_x.shape[0] > 1:
            use_x_last1 = torch.cat((self.last1_usex,use_x[:-1,:]),dim=0)
            use_x_last2 = torch.cat((self.last2_usex,use_x_last1[:-1, :]), dim=0)
            use_x_last3 = torch.cat((self.last3_usex,use_x_last2[:-1,:]),dim=0)
            use_x_last4 = torch.cat((self.last4_usex,use_x_last3[:-1, :]), dim=0)
        elif use_x.shape[0] == 1:
            use_x1_last1 = self.last_data['last1_usex1']
            use_x1_last2 = self.last_data['last2_usex1']
        self.last1_usex = use_x[-1, :].unsqueeze(0)
        self.last2_usex = use_x[-2, :].unsqueeze(0)
        self.last3_usex = use_x[-3, :].unsqueeze(0)
        self.last4_usex = use_x[-4, :].unsqueeze(0)

        use_x = torch.cat((use_x_last4.unsqueeze(2),use_x_last3.unsqueeze(2),use_x_last2.unsqueeze(2),use_x_last1.unsqueeze(2),use_x.unsqueeze(2)),dim=2)

        y1 = self.tcn1(use_x)#[N,C_out,L_out=L_in]
        compressor_speed = self.linear1(y1[:, :, -1])

        y2 = self.tcn2(use_x)#[N,C_out,L_out=L_in]
        cab_heating_status_act_pos = self.linear2(y2[:, :, -1])

        # 结果输出限幅
        if not self.training:
            # compressor_speed = torch.round(compressor_speed).int()
            cab_heating_status_act_pos = torch.round(cab_heating_status_act_pos).int()
            compressor_speed = torch.clamp(compressor_speed,max=self.compressor_speed_max,min=self.compressor_speed_min)
            cab_heating_status_act_pos = torch.clamp(cab_heating_status_act_pos,max=self.cab_heating_status_act_pos_max,min=self.cab_heating_status_act_pos_min)
            cab_heating_status_act_pos = torch.round(cab_heating_status_act_pos).int()
        return compressor_speed,cab_heating_status_act_pos


if __name__ == '__main__':
    model_params = {
        # 'input_size',C_in
        'input_size': 6,
        # 单步，预测未来一个时刻
        'output_size': 2,
        'num_channels': [64] * 2,
        'kernel_size': 3,
        'dropout': .0
    }
    net = tcn_net(**model_params)

    x = torch.randn([1024,9,1])
    y = net(x)
    # x = torch.randn([2,6,3])
    # print(x)
    # net = nn.Conv1d(6, 2, 3,
    #           stride=1, padding=2, dilation=1)
    #
    # y = net(x)
    #
    # print(y)