import copy
from json import load
from re import S
from turtle import forward
from torch.distributions import Normal
import torch.nn.functional as F
from click import Parameter
import torch
import torch.nn as nn
import torch.nn.init as init
from scipy.interpolate import interp1d

from AC_energy_pred.model.model_format import inter2D
import numpy as np
from scipy import interpolate
from AC_energy_pred.data_utils import get_device,cal_h
import AC_energy_pred.config_all as config
import pandas as pd

# 插值
def interp_from_table(data, table_key, table_value):
    # 创建插值函数
    interp_func = interp1d(table_key, table_value, kind='linear')

    # 进行插值
    interpolated_data = interp_func(data)

    return interpolated_data

def my_relu(input_tensor):
    # 创建一个与输入张量形状相同的全零张量
    zero_tensor = torch.zeros_like(input_tensor)
    # 应用 max 函数，比较每个元素并返回较大的那个
    return torch.max(zero_tensor, input_tensor)

def cal_temp_p_h_2(self,compressor_speed, hi_pressure, lo_pressure, temp_p_h_1_cab_heating):
    # 计算压比
    compressor_speed_np = compressor_speed
    pressure_ratio_np = hi_pressure / lo_pressure
    df = pd.read_excel('./model/GMCC涡旋压缩机Compressor Template 40&33cc R134a&R1234yf.xlsx',sheet_name='R134A-33CC')
    # 获取表格数据
    data_rpm = torch.tensor([i for i in range(2000,12001,1000)])
    data_pressure_ratio = torch.tensor([i for i in range(3, 9, 1)])
    data_volumetric_efficiency = torch.tensor(df['Efficiency'].values[2:].astype(float).reshape(len(data_rpm), len(data_pressure_ratio)))
    data_isentropic_efficiency = torch.tensor(df['Unnamed: 3'].values[2:].astype(float).reshape(len(data_rpm), len(data_pressure_ratio)))
    # 插值获取容积效率、等熵压缩效率
    f_interp_ve = inter2D(data_rpm, data_pressure_ratio, data_volumetric_efficiency,compressor_speed_np, pressure_ratio_np)
    f_interp_ie = inter2D(data_rpm, data_pressure_ratio, data_isentropic_efficiency,compressor_speed_np, pressure_ratio_np)
    # 计算2位置的理论焓和真实焓
    temp_p_h_1_cab_heating = temp_p_h_1_cab_heating
    lo_pressure = lo_pressure
    hi_pressure = hi_pressure
    h_1 = cal_h(temp_p_h_1_cab_heating.numpy(), lo_pressure.numpy(), states='gas')
    temp_h_2_real = (temp_p_h_1_cab_heating + 273.15) * (pressure_ratio_np ** (1 - 1/self.k)) - 273.15
    h_2_real = cal_h(temp_h_2_real.numpy(), hi_pressure.numpy(), states='gas')
    h_2_true = (h_2_real - h_1) / f_interp_ie + h_1
    # 通过反向查表得到2点温度
    p_h_x_log_p = config.p_h_x_log_p
    p_h_x_h = config.p_h_t_h
    p_h_x_t = config.p_h_t_t
    # temp_h_2 = inter2D(p_h_x_h, p_h_x_log_p, p_h_x_t, h_2_true, np.log(hi_pressure,dtype=np.float64))
    temp_h_2 = inter2D(p_h_x_log_p, p_h_x_h, p_h_x_t, np.log(lo_pressure,dtype=np.float64), h_2_true)
    return torch.tensor(temp_h_2)


class MLPModel_all(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 16)
        self.linear4 = torch.nn.Linear(16, 3)
        self.bn0 = torch.nn.BatchNorm1d(8)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(16)
        self.relu = torch.nn.ReLU()
        self.dp0 = torch.nn.Dropout(0.5)
        self.k = torch.nn.Parameter(torch.tensor([-0.1]),requires_grad=True)
        self.linear5 = torch.nn.Linear(1,1)


        self.df_odl = pd.read_excel('./model_ac/GMCC涡旋压缩机Compressor Template 40&33cc R134a&R1234yf.xlsx',sheet_name='R134A-33CC')
        # self.df = torch.from_numpy(pd.read_excel('./model/GMCC涡旋压缩机Compressor Template 40&33cc R134a&R1234yf.xlsx',sheet_name='R134A-33CC').to_numpy()).float()
            # 获取表格数据
        self.data_rpm = torch.tensor([i for i in range(2000,12001,1000)])
        self.data_pressure_ratio = torch.tensor([i for i in range(3, 9, 1)])
        # self.data_volumetric_efficiency = torch.tensor(self.df_odl['Efficiency'].values[2:].astype(float).reshape(len(self.data_rpm), len(self.data_pressure_ratio)))
        self.data_isentropic_efficiency = torch.tensor(self.df_odl['Unnamed: 3'].values[2:].astype(float).reshape(len(self.data_rpm), len(self.data_pressure_ratio)))
            # 通过反向查表得到2点温度
        self.p_h_t_log_p = torch.tensor(config.p_h_t_log_p)
        self.p_h_t_h = torch.tensor(config.p_h_t_h)
        self.p_h_t_t = torch.tensor(config.p_h_t_t)

    def forward(self, x):
        x0 = torch.cat((x[:, :6], x[:, 7].unsqueeze(1), x[:, 9].unsqueeze(1)), dim=1)
        compressor_speed = x[:, 2].unsqueeze(1)
        lo_pressure = x[:, 5].unsqueeze(1)
        temp_p_h_1_cab_heating = x[:, 4].unsqueeze(1)
        
        x1 = self.bn0(x0)
        x2 = self.relu(self.linear1(x1))
        x3 = self.bn1(x2)
        # x3 = self.dp0(x3)
        x4 = self.relu(self.linear2(x3))
        x5 = self.bn2(x4)
        # x5 = self.dp0(x5)
        x6 = self.relu(self.linear3(x5))
        x7 = self.bn3(x6)
        x8 = self.linear4(x7)
        cc = x8[:,1].unsqueeze(1)
        hi_pressure = x8[:, 2].unsqueeze(1).clone().detach()
        # 保证 hi_pressure >= lo_pressure
        hi_pressure = torch.max(hi_pressure, lo_pressure)
        
        pressure_ratio = hi_pressure / lo_pressure
        f_interp_ie = inter2D(self.data_rpm, self.data_pressure_ratio, self.data_isentropic_efficiency, compressor_speed[:, 0], pressure_ratio[:, 0])
        h_1 = torch.from_numpy(cal_h(temp_p_h_1_cab_heating.detach().numpy(), lo_pressure.detach().numpy(), states='gas')).float()
        temp_h_2_real = (temp_p_h_1_cab_heating + 273.15) * (pressure_ratio ** self.k) - 273.15
        h_2_real = torch.from_numpy(cal_h(temp_h_2_real.detach().numpy(), hi_pressure.detach().numpy(), states='gas')).float()
        h_2_true = (h_2_real - h_1) / f_interp_ie + h_1
        temp_h_2 = inter2D(self.p_h_t_log_p, self.p_h_t_h, self.p_h_t_t, torch.log(hi_pressure[:, 0]), h_2_true).unsqueeze(1)
        # temp_h_2 = self.linear5(temp_h_2)
        temp_h_2 = temp_h_2 + 0.1*cc
        return torch.cat((x8[:, 0].unsqueeze(1), temp_h_2, x8[:, 2].unsqueeze(1)), dim=1)

class MLPModel_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128,64)
        self.linear4 = torch.nn.Linear(64, 3)
        self.bn0 = nn.BatchNorm1d(8)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        # self.dp1 = nn.Dropout(0.05)
        # self.dp2 = nn.Dropout(0.02)
        self.relu = my_relu
        self.gelu = nn.GELU()
        self.k1 = nn.Parameter(torch.tensor(1.0))
        self.k2 = nn.Parameter(torch.tensor(1.0))
        self.k3 = nn.Parameter(torch.tensor(1.0))
        self.k4 = nn.Parameter(torch.tensor(1.0))
        self.k5 = nn.Parameter(torch.tensor(1.0))
        self.k6 = nn.Parameter(torch.tensor(1.0))
        self.k7 = nn.Parameter(torch.tensor(1.0))
        self.k8 = nn.Parameter(torch.tensor(0.6))

    def forward(self, x):
        x0 = torch.cat((x[:, :6], x[:, 7].unsqueeze(1),  x[:, 9].unsqueeze(1)), dim=1).clone()
        x1 = self.bn0(x0)
        h1 = self.relu(self.bn1(self.linear1(x1)))
        # h1 = self.dp1(h1)
        h2 = self.relu(self.bn2(self.linear2(h1)))
        # h2 = self.dp2(h2)
        h3 = self.relu(self.bn3(self.linear3(h2)))
        out = self.linear4(h3)
        
        # out2 = out[:,1].clone()
        temp_p_h_1_cab_heating = (x[:,4] + 273.15).clone().detach() # 正相关
        hi_pressure = out[:,2].clone()
        low_pressure = x[:,5].clone().detach() # 正相关
        speed = x[:,2].clone().detach() # 正相关
        cab_heating_status_act_pos = (x[:,3]+0).clone().detach() # 负相关
        air_temp_before_heat_exchange = (x[:,0] + 273.15).clone().detach() # 正相关
        wind_vol = x[:,1].clone().detach() # 正相关
        hvch_cool_medium_temp_out = (x[:,9] + 273.15).clone().detach() # 正相关

        # hi_pressure = torch.max(hi_pressure, low_pressure)
        k1 = self.relu(self.k1)
        k2 = self.relu(self.k2)
        k3 = self.relu(self.k3)
        k4 = -self.relu(self.k4)
        k5 = self.relu(self.k5)
        k6 = self.relu(self.k6)
        k7 = self.relu(self.k7)
        k8 = self.relu(self.k8)
        # b1 = self.relu(self.b1)
        out3 = k1 * temp_p_h_1_cab_heating + k2 * low_pressure + k3 * speed \
            + k4 * cab_heating_status_act_pos + k5 * air_temp_before_heat_exchange \
            + k6 * wind_vol + k7 * hvch_cool_medium_temp_out + k8 * hi_pressure
        out4 = torch.max(out3, temp_p_h_1_cab_heating)
        out[:,1] = out[:,1] + out4
        # out[:,2] = hi_pressure
        return out

def limit_lin(x):

    if (torch.abs(x) > 65535).any():
        x = torch.clamp(x, min=-65504, max=65504)

    # x = torch.clamp(x * 1e3, min=0)
    # x = x / 1e3
    return x

def limit_k(param):
    with torch.no_grad():
        # 保留四位小数
        # param = torch.clamp(param * 1e3, min=0)
        # param = param / 1e3
        # 裁剪参数
        param = torch.clamp(param, -65504, 65504)
    return param



# # model1
class MLPModel_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128,16)
        self.linear4 = torch.nn.Linear(16, 3)
        self.bn0 = nn.BatchNorm1d(8)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(16)
        # self.dp1 = nn.Dropout(0.05)
        # self.dp2 = nn.Dropout(0.02)
        self.relu = my_relu
        self.gelu = nn.GELU()
        self.k1 = nn.Parameter(torch.tensor(0.99))
        self.k2 = nn.Parameter(torch.tensor(1.01))
        self.k3 = nn.Parameter(torch.tensor(1.1))
        self.k4 = nn.Parameter(torch.tensor(1.0))
        self.k5 = nn.Parameter(torch.tensor(0.9))
        # self.b1 = nn.Parameter(torch.tensor(273.15))
    def forward(self, x):

        x0 = torch.cat((x[:, :6], x[:, 7].unsqueeze(1),  x[:, 9].unsqueeze(1)), dim=1).clone()
        x1 = self.bn0(x0)
        h1 = self.relu(self.bn1(self.linear1(x1)))
        h1 = limit_lin(h1)
        # h1 = self.dp1(h1)
        h2 = self.relu(self.bn2(self.linear2(h1)))
        h2 = limit_lin(h2)
        # h2 = self.dp2(h2)
        h3 = self.relu(self.bn3(self.linear3(h2)))
        h3 = limit_lin(h3)
        out = self.linear4(h3)
        out = limit_lin(out)
        # out2 = out[:,1].clone()
        temp_p_h_1_cab_heating = (x[:,4] + 273.15).clone()
        hi_pressure = out[:,2].clone()
        low_pressure = x[:,5].clone()
        speed = x[:,2].clone()

        for i in range(x.shape[0]):
            if hi_pressure[i] < low_pressure[i]:
                hi_pressure[i] = low_pressure[i]
        k3 = self.relu(self.k3)
        k1 = self.relu(self.k1)
        k2 = self.relu(self.k2)
        k4 = self.relu(self.k4)
        k5 = self.relu(self.k5)
        k3 = limit_k(k3)
        k1 = limit_k(k1)
        k2 = limit_k(k2)
        k4 = limit_k(k4)
        k5 = limit_k(k5)

        # b1 = self.relu(self.b1)
        out3 = k3*(temp_p_h_1_cab_heating**k4)*(1+k1 * ((hi_pressure/low_pressure)**(k2/(1+k2)) -1))*(speed/3600)**k5 - 273.15
        out4 = torch.relu(out3 - x[:,4]) + x[:,4]
        out[:,1] = out4
        return out
