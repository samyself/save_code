import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d

# import config


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=[256, 128]):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        # self.linear3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.out_linear = nn.Linear(hidden_dim[1], 1)

        self.bn = torch.nn.BatchNorm1d(hidden_dim[0])

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = torch.relu(x1)
        x1 = self.bn(x1)
        x2 = self.linear2(x1)
        out = self.out_linear(x2)

        return out


# 制冷模型
class WindTempModel_cool_phsyics(nn.Module):
    def __init__(self):
        super().__init__()

        # 物态变化第一阶段参数 冷凝剂为气液混合态
        self.lambda_1 = nn.Parameter(torch.tensor(7.5e1))

        # 物态变化第二阶段参数 冷凝剂为气态
        self.lambda_2 = nn.Parameter(torch.tensor(3e2))
        self.lambda_3 = nn.Parameter(torch.tensor(5e2))
        # self.t_1 = nn.Parameter(torch.tensor(1e-2))

    def forward(self, **kwargs):
        # hp_mode=kwargs['hp_mode']
        refrigerant_mix_temp = kwargs['refrigerant_mix_temp']
        refrigerant_air_temp = kwargs['refrigerant_air_temp']
        wind_vol = kwargs['wind_vol']
        air_temp_before_heat_exchange = kwargs['air_temp_before_heat_exchange']
        cab_cooling_status_act_pos = kwargs['cab_cooling_status_act_pos']

        # 管道长度和横截面积
        channel_len = config.channel_cooling_place
        channel_s = config.channel_s

        # 根据长度和横截面积计算气流换热时间
        total_time = channel_len / (wind_vol / channel_s)

        batch_size = len(refrigerant_mix_temp)
        '''物态变化第一阶 冷凝剂由气态转化为气液混合状态'''
        lambda_1 = torch.relu(self.lambda_1)

        # 初始的风温和制冷剂温度
        t_w_0 = air_temp_before_heat_exchange
        t_r_0 = refrigerant_mix_temp

        t_1 = total_time
        pred_t_w_1 = (t_w_0 - t_r_0) * torch.exp(-lambda_1 * t_1) + t_r_0

        '''物态变化第二阶 冷凝剂保持气液混合状态到气态'''
        lambda_2 = torch.relu(self.lambda_2)
        lambda_3 = torch.relu(self.lambda_3)
        # t_2 = 1e-5 * total_time
        t_2 = 0 * total_time

        # 初始的风温和制冷剂温度
        t_w_1 = pred_t_w_1
        t_r_1 = t_r_0

        step_2_para_b_0 = t_r_1
        step_2_para_b_1 = t_w_1

        step_2_para_A_0_0 = 1
        step_2_para_A_0_1 = 1
        step_2_para_A_1_0 = -lambda_2 / lambda_3
        step_2_para_A_1_1 = 1

        step_2_para_A = torch.tensor([[step_2_para_A_0_0, step_2_para_A_0_1], [step_2_para_A_1_0, step_2_para_A_1_1]],
                                     device=step_2_para_A_1_0.device)
        step_2_para_A = step_2_para_A.unsqueeze(0).repeat(batch_size, 1, 1)
        step_2_para_b = torch.cat([step_2_para_b_0.view(-1, 1), step_2_para_b_1.view(-1, 1)], dim=1)

        step_2_solutions = torch.linalg.solve(step_2_para_A, step_2_para_b)
        step_2_c_1 = step_2_solutions[:, 0]
        step_2_c_2 = step_2_solutions[:, 1]

        pred_t_r_2 = step_2_para_A_0_0 * step_2_c_1 * torch.exp(-(lambda_2 + lambda_3) * t_2) + step_2_c_2
        pred_t_w_2 = step_2_para_A_1_0 * step_2_c_1 * torch.exp(-(lambda_2 + lambda_3) * t_2) + step_2_c_2

        return pred_t_r_2, pred_t_w_2


# 加热模型
class WindTempModel_heat_phsyics(nn.Module):
    def __init__(self):
        super().__init__()

        # 物态变化第一阶段参数 冷凝剂为气态
        self.lambda_1 = nn.Parameter(torch.tensor(3e2))
        self.lambda_2 = nn.Parameter(torch.tensor(5e2))
        # self.t_1 = nn.Parameter(torch.tensor(1e-2))

        # 物态变化第二阶 冷凝剂保持气液混合状态
        self.lambda_3 = nn.Parameter(torch.tensor(1e2))
        # self.t_2 = nn.Parameter(torch.tensor(1e-1))

        # 物态变化第三阶 冷凝剂为液态
        self.lambda_4 = nn.Parameter(torch.tensor(1e2))
        self.lambda_5 = nn.Parameter(torch.tensor(1e1))
        # self.t_3 = nn.Parameter(torch.tensor(1e-2))

    def forward(self, **kwargs):
        # hp_mode=kwargs['hp_mode']
        step_1_refrigerant_temp = kwargs['step_1_refrigerant_temp']
        step_2_refrigerant_temp = kwargs['step_2_refrigerant_temp']
        wind_vol = kwargs['wind_vol']
        air_temp_before_heat_exchange = kwargs['air_temp_before_heat_exchange']
        cab_heating_status_act_pos = kwargs['cab_heating_status_act_pos']

        # 管道长度和横截面积
        channel_len = config.channel_heating_place
        channel_s = config.channel_s

        # 根据长度和横截面积计算气流换热时间
        total_time = channel_len / (wind_vol / channel_s)

        batch_size = len(step_1_refrigerant_temp)
        '''物态变化第一阶 冷凝剂由气态转化为气液混合状态'''
        lambda_1 = torch.relu(self.lambda_1)
        lambda_2 = torch.relu(self.lambda_2)
        # t_1 = torch.relu(self.t_1)

        # 初始的风温和制冷剂温度
        t_w_0 = air_temp_before_heat_exchange
        t_r_0 = step_1_refrigerant_temp

        t_r_1 = step_2_refrigerant_temp

        step_1_para_b_0 = t_r_0
        step_1_para_b_1 = t_w_0

        step_1_para_A_0_0 = 1
        step_1_para_A_0_1 = 1
        step_1_para_A_1_0 = -lambda_1 / lambda_2
        step_1_para_A_1_1 = 1

        step_1_para_A = torch.tensor([[step_1_para_A_0_0, step_1_para_A_0_1], [step_1_para_A_1_0, step_1_para_A_1_1]],
                                     device=step_1_para_A_1_0.device)
        step_1_para_A = step_1_para_A.unsqueeze(0).repeat(batch_size, 1, 1)
        step_1_para_b = torch.cat([step_1_para_b_0.view(-1, 1), step_1_para_b_1.view(-1, 1)], dim=1)

        step_1_solutions = torch.linalg.solve(step_1_para_A, step_1_para_b)
        step_1_c_1 = step_1_solutions[:, 0]
        step_1_c_2 = step_1_solutions[:, 1]

        # pred_t_r_1 = step_1_para_A_0_0 * step_1_c_1 * torch.exp(-(lambda_1 + lambda_2) * t_1) + step_1_c_2
        # 直接计算时间
        t_1 = torch.log((torch.relu((t_r_1 - step_1_c_2) / (step_1_para_A_0_0 * step_1_c_1))) + 1e-10) / (-(lambda_1 + lambda_2))
        t_1 = torch.relu(t_1)
        pred_t_w_1 = step_1_para_A_1_0 * step_1_c_1 * torch.exp(-(lambda_1 + lambda_2) * t_1) + step_1_c_2

        remain_time = torch.relu(total_time - t_1)

        '''物态变化第二阶 冷凝剂保持气液混合状态'''
        lambda_3 = torch.relu(self.lambda_3)
        # t_2 = torch.relu(self.t_2)
        time_weight_step_2 = 0.1
        t_2 = remain_time * time_weight_step_2
        pred_t_w_2 = (pred_t_w_1 - t_r_1) * torch.exp(-lambda_3 * t_2) + t_r_1

        '''物态变化第一阶 冷凝剂由气液混合状态化为液态'''
        lambda_4 = torch.relu(self.lambda_4)
        lambda_5 = torch.relu(self.lambda_5)
        # t_3 = torch.relu(self.t_3)
        t_3 = remain_time * (1 - time_weight_step_2)

        # 初始的风温和制冷剂温度
        t_w_2 = pred_t_w_2
        t_r_2 = t_r_1

        step_3_para_b_0 = t_r_2
        step_3_para_b_1 = t_w_2

        step_3_para_A_0_0 = 1
        step_3_para_A_0_1 = 1
        step_3_para_A_1_0 = -lambda_4 / lambda_5
        step_3_para_A_1_1 = 1

        step_3_para_A = torch.tensor([[step_3_para_A_0_0, step_3_para_A_0_1], [step_3_para_A_1_0, step_3_para_A_1_1]],
                                     device=step_3_para_A_1_0.device)
        step_3_para_A = step_3_para_A.unsqueeze(0).repeat(batch_size, 1, 1)
        step_3_para_b = torch.cat([step_3_para_b_0.view(-1, 1), step_3_para_b_1.view(-1, 1)], dim=1)

        step_3_solutions = torch.linalg.solve(step_3_para_A, step_3_para_b)
        step_3_c_1 = step_3_solutions[:, 0]
        step_3_c_2 = step_3_solutions[:, 1]

        pred_t_r_3 = step_3_para_A_0_0 * step_3_c_1 * torch.exp(-(lambda_4 + lambda_5) * t_3) + step_3_c_2
        pred_t_w_3 = step_3_para_A_1_0 * step_3_c_1 * torch.exp(-(lambda_4 + lambda_5) * t_3) + step_3_c_2

        return pred_t_r_3, pred_t_w_3
