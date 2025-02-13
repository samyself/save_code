import torch
import torch.nn as nn
import torch.nn.init as init
from model.data_utils_common import inter2D,inter1D


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, use_batchnorm=True, use_dropout=True):
        super(MLP, self).__init__()
        self.use_batchnorm = use_batchnorm
        layers = []

        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())

        # 添加额外的隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
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
    def __init__(self, car_type='modena'):
        super().__init__()
        self.car_type = car_type

        # CabinP1_table_modena[:,0]是环境温度temp_amb， CabinP1_table_modena[:,1]是CabinP1
        self.CabinP1_table_modena = torch.tensor([[-20.0, 3.0], [-10.0, 2.0], [0.0, 0.88], [10.0, 0.38], [25.0, 0.7], [30.0, 1.0], [35.0, 1.31], [40.0, 2.5], [45.0, 3.0]])
        self.CabinP1_table_lemans = torch.tensor([[-20.0, 3.0], [-10.0, 2.0], [0.0, 0.875], [10.0, 0.375], [25.0, 0.69531], [30.0, 1.0], [35.0, 1.3125], [40.0, 2.5], [45.0, 3.0]])

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
                                        [32, 32, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36]])            # 32

        self.CabinP2_table = torch.tensor(
            [[-20.0, 3.0], [-15.0, 2.0], [-10.0, 1.0], [-5.0, 0.50], [0.0, 0.40], [5.0, 0.50], [10.0, 1.5],
             [15.0, 3.0], [20.0, 6.0]])

        # self.ChillerP1_table_modena = torch.tensor([[-10.0,4.0], [-5.0,2.0], [-1.0,1.0], [0.0,0.40], [1.0,0.40], [2.0,0.40], [5.0,1.0], [10.0,2.0], [20.0,3.0], [30.0,4.0]])
        # self.ChillerP1_table_lemans = torch.tensor(
        #     [[-10.0, 6.0], [-5.0, 3.0], [-1.0, 1.5], [0.0, 0.79980], [1.0, 0.79980], [2.0, 1.20019], [5.0, 1.5], [10.0, 3.0],
        #      [20.0, 4.5], [30.0, 6.0]])

        self.MLP1 = MLP(input_size=1, hidden_sizes=[64, 64], output_size=1,use_batchnorm=False,use_dropout=False)
        # self.MLP2 = MLP(input_size=4, hidden_sizes=[64, 64], output_size=1, use_batchnorm=False, use_dropout=False)
        # self.bais = nn.Parameter(torch.rand(1), requires_grad=True)

        init_weights_xavier_uniform(self.MLP1)
        self.k = nn.Parameter(torch.rand(1), requires_grad=True)
        self.b = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x):
        # 压缩机排气温度、内冷温度、饱和高压、压缩机进气温度、饱和低压、目标饱和高压、目标过冷度、目标过热度
        # 压缩机转速、膨胀阀开度
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        # 输出限制
        # 当前环境温度
        temp_amb = x[:, 1]

        # 当前主驾驶设定温度
        cab_fl_set_temp = x[:,2]
        # 当前副驾驶设定温度
        cab_fr_set_temp = x[:,3]

        # 当前乘务舱温度
        temp_incar = x[:,8]
        # 电池请求冷却液温度
        temp_battery_req = x[:,9]
        # 冷却液进入电池的温度
        temp_coolant_battery_in = x[:,10]

        if 'modena' in self.car_type:
            CabinP1_table = self.CabinP1_table_modena
        elif 'lemans' in self.car_type:
            CabinP1_table = self.CabinP1_table_lemans

        AC_KpRateCabin1 = inter1D(CabinP1_table[:,0], CabinP1_table[:,1], temp_amb)

        #   DVT_CabinErr_FL = f(主驾设定温度, 环境温度) 查表 - SEN_Incar(乘员舱温度)
        DVT_CabinErr_FL = inter2D(self.temp_set,self.temp_envr,self.CabinSP_table,cab_fl_set_temp,temp_amb) - temp_incar.view(-1,1)
        #   DVT_CabinErr_FR = f(副驾设定温度, 环境温度) 查表 - SEN_Incar(乘员舱温度)
        DVT_CabinErr_FR = inter2D(self.temp_set, self.temp_envr, self.CabinSP_table,cab_fr_set_temp, temp_amb) - temp_incar.view(-1,1)

        DVT_CabinErr = torch.min(DVT_CabinErr_FL, DVT_CabinErr_FR)

        AC_KpRateCabin2 = inter1D(self.CabinP2_table[:,0], self.CabinP2_table[:,1], DVT_CabinErr)

        AC_KpRateCabin = torch.min(AC_KpRateCabin1, AC_KpRateCabin2)


        # DCT_DCT_ChillerTempErr = HP_Mode_BattWTReqOpt（电池请求冷却的温度） - SEN_CoolT_Battln（冷却液进入电池的温度）
        # DCT_ChillerTempErr  = temp_battery_req - temp_coolant_battery_in
        #
        # if 'modena' in self.car_type:
        #     ChillerP1_table = self.ChillerP1_table_modena
        # else:
        #     ChillerP1_table = self.ChillerP1_table_lemans
        #
        # AC_KpRateChiller = inter1D(ChillerP1_table[:,0], ChillerP1_table[:,1], DCT_ChillerTempErr)

        # use_x1 = torch.cat((AC_KpRateCabin.view(-1,1), AC_KpRateChiller.view(-1,1)), dim=1)
        # use_x2 = torch.cat((AC_KpRateCabin.view(-1,1), AC_KpRateChiller.view(-1,1), DVT_CabinErr_FL.view(-1,1), DVT_CabinErr_FR.view(-1,1)), dim=1)

        # AC_KpRate = self.MLP1(use_x1)

        AC_KpRate = AC_KpRateCabin
        return AC_KpRate

if __name__  == '__main__':

    temp_set = torch.tensor([18.0, 20, 22, 24, 26, 28, 30, 31.5, 32])
    # 环境温度
    temp_envr = torch.tensor([-30.0, -20, -10, 0, 5, 10, 15, 20, 25])


    print(inter1D(temp_set, temp_envr, torch.tensor([17.0,34])))