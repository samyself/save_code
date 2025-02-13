import os
from mpl_toolkits.mplot3d import Axes3D
import torch
from matplotlib import pyplot as plt
from torch import nn
from model.model_ac_pid_out_PID import MyModel

from model import *


def get_aim_data(data_name, data_dict):
    data = []
    for para_index in range(len(data_name)):
        input_name = data_name[para_index]
        data.append(data_dict[input_name])

    data = torch.cat(data, dim=-1)
    return data


def get_data_dict(data_name, data):
    if isinstance(data, tuple):
        data = torch.cat(data, dim=-1)

    if len(data.shape) == 1:
        data = data.view(-1, 1)

    data_dict = {}
    for para_index in range(len(data_name)):
        name = data_name[para_index]
        data_dict[name] = data[:, para_index].view(-1, 1)

    return data_dict

model_dict = {'ACControlModel': MyModel()}
sub_model_ckpt_path_dict = {'ACControlModel': './data/ckpt/ac_1021_v5.pth'}
model_input_name_dict = {'ACControlModel': ['lst_ac_pid_out_hp', 'temp_amb', 'cab_fl_set_temp', 'cab_fr_set_temp',
                                            'lo_pressure', 'hi_pressure', 'aim_lo_pressure', 'aim_hi_pressure',
                                            'temp_incar', 'temp_battery_req', 'temp_coolant_battery_in', 'ac_kp_rate',
                                            'temp_p_h_2', 'temp_p_h_1_cab_heating', 'temp_p_h_5']}
model_output_name_dict = {'ACControlModel': ['ac_pid_out_hp']}

if __name__ == '__main__':
    aim_model_name = 'ACControlModel'
    net = model_dict[aim_model_name]
    net.eval()
    ckpt_path = sub_model_ckpt_path_dict[aim_model_name]
    # optimizer = optim.SGD(net.parameters(), lr=config_all.lr)
    if os.path.exists(ckpt_path):
        net.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))

    pic_path_folder = './data/correlation'
    if not os.path.exists(pic_path_folder):
        os.makedirs(pic_path_folder)

    # 目标字段
    input_name_list = model_input_name_dict[aim_model_name]
    aim_check_para_name_list = ['aim_hi_pressure']

    '''根据范围确定采样空间'''
    temp_p_h_2_max = 115.0
    temp_p_h_2_min = 40.0
    temp_p_h_5_max = 65.0
    temp_p_h_5_min = 20.0
    hi_pressure_max = 2400.0
    hi_pressure_min = 900.0
    temp_p_h_1_cab_heating_max = 31.0
    temp_p_h_1_cab_heating_min = -18.0
    lo_pressure_max = 700.0
    lo_pressure_min = 100.0
    aim_hi_pressure_max = 2400.0
    aim_hi_pressure_min = 900.0
    ac_pid_out_hp_max = 8000.0
    ac_pid_out_hp_min = 0
    temp_amb_max = 55.0
    temp_amb_min = -40.0
    cab_set_temp_max = 32.0
    cab_set_temp_min = 18.0
    cab_req_temp_max = 60.0
    cab_req_temp_min = -30.0

    range_dict = {
        'lst_ac_pid_out_hp': [ac_pid_out_hp_min, ac_pid_out_hp_max],
        'temp_amb':[temp_amb_min, temp_amb_max],
        'cab_fl_set_temp':[cab_set_temp_min, cab_set_temp_max],
        'cab_fr_set_temp':[cab_set_temp_min, cab_set_temp_max],
        'lo_pressure': [lo_pressure_min, lo_pressure_max],
        'hi_pressure': [hi_pressure_min, hi_pressure_max],
        'aim_lo_pressure': [lo_pressure_min, lo_pressure_max],
        'aim_hi_pressure': [aim_hi_pressure_min, aim_hi_pressure_max],
        'temp_incar': [-30.0, 65],
        'temp_battery_req': [10.0, 45.0],
        'temp_coolant_battery_in': [10.0, 45.0],
        'ac_kp_rate': [0.0,4.0],
        'temp_p_h_2': [temp_p_h_2_min, temp_p_h_2_max],
        'temp_p_h_1_cab_heating': [temp_p_h_1_cab_heating_min, temp_p_h_1_cab_heating_max],
        'temp_p_h_5': [temp_p_h_5_min, temp_p_h_5_max]
    }
    # 每个通道采样个数
    sample_num_per_input = 2
    opt_name_range_list = []
    for para_index in range(len(input_name_list)):
        now_input_name = input_name_list[para_index]
        now_range = range_dict[now_input_name]
        min_data = now_range[0]
        max_data = now_range[1]
        split_len = (max_data - min_data) / (sample_num_per_input - 1)
        if split_len == 0:
            split_len = 1
        range_data = torch.arange(min_data, max_data + 1, split_len)
        opt_name_range_list.append(range_data)

    # 组合为新数据
    meshgrid_out = torch.meshgrid(opt_name_range_list)
    meshgrid_prob_comb = torch.cat([meshgrid_out[i].unsqueeze(-1) for i in range(len(meshgrid_out))], dim=-1)
    prob_comb = meshgrid_prob_comb.view(-1, meshgrid_prob_comb.shape[-1])
    input_data_dict = get_data_dict(model_input_name_dict[aim_model_name], prob_comb)

    # 对于要考虑相关性的字段 需要考虑在每个点周围改变多少
    # dx_rate = 1e-2
    final_input_data_dict = {}
    # final_input_data_dict_p = {}
    # final_input_data_dict_d = {}

    for para_index in range(len(input_name_list)):
        now_input_name = input_name_list[para_index]
        now_data = input_data_dict[now_input_name]

        if now_input_name in aim_check_para_name_list:
            now_data = now_data.clone().detach().requires_grad_(True)
            # 计算附近情况
            # now_data_dx_p = now_data.clone().detach().requires_grad_(True) * (dx_rate + 1)
            # now_data_dx_d = now_data.clone().detach().requires_grad_(True) * (-dx_rate + 1)
        else:
            now_data = now_data.clone().detach()
            # 计算附近情况
            # now_data_dx_p = now_data.clone().detach()
            # now_data_dx_d = now_data.clone().detach()

        final_input_data_dict[now_input_name] = now_data
        # final_input_data_dict_p[now_input_name] = now_data_dx_p
        # final_input_data_dict_d[now_input_name] = now_data_dx_p

    final_input = get_aim_data(input_name_list, final_input_data_dict)
    # final_input_p = get_aim_data(input_name_list, final_input_data_dict_p)
    # final_input_d = get_aim_data(input_name_list, final_input_data_dict_d)

    out = torch.zeros([final_input.shape[0], 1])
    for i in range(len(final_input)):
        if i % 30 == 0:
            net.last_ac_pid_out_hp = final_input[i,0]
        out[i] = net(final_input[i,:])
    # out = net(final_input)
    # out_p = net(final_input_p)
    # out_d = net(final_input_d)
    result_dict = get_data_dict(model_output_name_dict[aim_model_name], out)
    # result_dict_p = get_data_dict(model_output_name_dict[aim_model_name], out_p)
    # result_dict_d = get_data_dict(model_output_name_dict[aim_model_name], out_d)
    compressor_speed = result_dict['ac_pid_out_hp']
    # compressor_speed_p = result_dict_p['compressor_speed']
    # compressor_speed_d = result_dict_d['compressor_speed']

    gradients = torch.ones_like(compressor_speed)
    aim_hi_pressure = final_input_data_dict['aim_hi_pressure']
    grad_out = torch.autograd.grad(compressor_speed, aim_hi_pressure, gradients, allow_unused=True, retain_graph=True,create_graph=True)[0]
    ok_rate = len(grad_out[grad_out > 0]) / len(out)
    print(ok_rate)

