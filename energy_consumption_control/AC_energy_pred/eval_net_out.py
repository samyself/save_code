import os
from mpl_toolkits.mplot3d import Axes3D
import torch
from matplotlib import pyplot as plt
from torch import nn
from model.model_cmpr_control_gai import MyModel

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
sub_model_ckpt_path_dict = {'ACControlModel': './data/ckpt/cc_1010_sub2.pth'}
model_input_name_dict = {'ACControlModel': ['last_temp_p_h_2', 'last_temp_p_h_5', 'last_hi_pressure', 'last_temp_p_h_1_cab_heating',
                                            'last_lo_pressure', 'aim_hi_pressure', 'aim_lo_pressure', 'sc_tar_mode_10', 'sh_tar_mode_10']}
model_output_name_dict = {'ACControlModel': ['compressor_speed','cab_heating_status_act_pos']}

if __name__ == '__main__':
    aim_model_name = 'ACControlModel'
    net = model_dict[aim_model_name]
    ckpt_path = sub_model_ckpt_path_dict[aim_model_name]
    if os.path.exists(ckpt_path):
        net.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))

    pic_path_folder = './data/net_out'
    if not os.path.exists(pic_path_folder):
        os.makedirs(pic_path_folder)

    input_name_list = model_input_name_dict[aim_model_name]

    aim_check_para_name_list = ['aim_lo_pressure']
    # aim_check_para_name_list = input_name_list
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

    range_dict = {
        'last_temp_p_h_2': [temp_p_h_2_min, temp_p_h_2_max],
        'last_temp_p_h_5': [temp_p_h_5_min, temp_p_h_5_max],
        'last_hi_pressure': [hi_pressure_min, hi_pressure_max],
        'last_temp_p_h_1_cab_heating': [temp_p_h_1_cab_heating_min, temp_p_h_1_cab_heating_max],
        'last_lo_pressure': [lo_pressure_min, lo_pressure_max],
        'aim_hi_pressure': [aim_hi_pressure_min, aim_hi_pressure_max],
        'aim_lo_pressure': [lo_pressure_min, lo_pressure_max],
        'sc_tar_mode_10': [0.0, 0.0],
        'sh_tar_mode_10': [0.0, 0.0]
    }

    sample_num_per_input = 5
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

    meshgrid_out = torch.meshgrid(opt_name_range_list)
    meshgrid_prob_comb = torch.cat([meshgrid_out[i].unsqueeze(-1) for i in range(len(meshgrid_out))], dim=-1)
    prob_comb = meshgrid_prob_comb.view(-1, meshgrid_prob_comb.shape[-1])
    input_data_dict = get_data_dict(model_input_name_dict[aim_model_name], prob_comb)
    final_input_data_dict = {}
    final_input_data_dict_p = {}
    final_input_data_dict_d = {}
    dx_rate = 1e-1
    for para_index in range(len(input_name_list)):
        now_input_name = input_name_list[para_index]
        now_data = input_data_dict[now_input_name]

        if now_input_name in aim_check_para_name_list:
            now_data = now_data.clone().detach().requires_grad_(True)
            # 计算附近情况
            now_data_dx_p = now_data.clone().detach().requires_grad_(True) * (dx_rate + 1)
            now_data_dx_d = now_data.clone().detach().requires_grad_(True) * (-dx_rate + 1)
        else:
            now_data = now_data.clone().detach().requires_grad_(True)
            # 计算附近情况
            now_data_dx_p = now_data.clone().detach().requires_grad_(True)
            now_data_dx_d = now_data.clone().detach().requires_grad_(True)
        final_input_data_dict[now_input_name] = now_data
        final_input_data_dict_p[now_input_name] = now_data_dx_p
        final_input_data_dict_d[now_input_name] = now_data_dx_p

    final_input = get_aim_data(input_name_list, final_input_data_dict)
    final_input_p = get_aim_data(input_name_list, final_input_data_dict_p)
    final_input_d = get_aim_data(input_name_list, final_input_data_dict_d)
    net.eval()
    out = net(final_input)
    out_p = net(final_input_p)
    out_d = net(final_input_d)
    result_dict = get_data_dict(model_output_name_dict[aim_model_name], out)
    result_dict_p = get_data_dict(model_output_name_dict[aim_model_name], out_p)
    result_dict_d = get_data_dict(model_output_name_dict[aim_model_name], out_d)
    compressor_speed = result_dict['compressor_speed']
    compressor_speed_p = result_dict_p['compressor_speed']
    compressor_speed_d = result_dict_d['compressor_speed']

    diff_p = compressor_speed - compressor_speed_p
    diff_d = compressor_speed - compressor_speed_d

    # 目标高压变大的时候 需要计算转速变大的比例
    ok_p_rate = len(torch.where(diff_p < 0)[0]) / len(out)
    # 目标高压变小的时候 需要计算转速变小的比例
    ok_d_rate = len(torch.where(diff_d > 0)[0]) / len(out)
    print('up',ok_p_rate,'down', ok_d_rate)

    # 若希望转速减少
    # criterion = nn.MSELoss()
    # loss = criterion(compressor_speed, torch.zeros_like(compressor_speed))
    # loss.backward()
    #
    # last_hi_pressure = final_input_data_dict['last_hi_pressure']
    # aim_hi_pressure = final_input_data_dict['aim_hi_pressure']
    #
    # # last_hi_pressure-aim_hi_pressure的导 是直接导数求差
    # aim_grad_base=last_hi_pressure-aim_hi_pressure
    # aim_grad_base_grad=last_hi_pressure.grad-aim_hi_pressure.grad

    # 若希望减少转速 那么需要降低高压 且降低目标高压 且需要拉大last_hi_pressure-aim_hi_pressure

    # gradients = torch.ones_like(loss)
    # out=torch.autograd.grad(loss, aim_grad_base,gradients, allow_unused=True,retain_graph=True,create_graph=True)

    # print(aim_grad_base_grad)

    # fig = plt.figure()  # 创建一个画布figure，然后在这个画布上加各种元素。
    # ax = Axes3D(fig)
    # ax.scatter(weight_input[:, 0], weight_input[:, 1], weight)
    # ax.set_xlabel('speed')
    # ax.set_ylabel('flowrate_oil')
    # ax.set_zlabel('weight')
    #
    # pic_path = os.path.join(pic_path_folder, f'{net_name}_data.png')
    # plt.savefig(pic_path)
    # plt.close()
