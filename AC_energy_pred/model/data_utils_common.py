import numpy as np
from scipy.interpolate import interp2d, interp1d

from common import config_common
import torch


# 单线性插值
# 输入x_list, y_list, x输入维度为b*1
# 输出y 为b*1的tensor
# def inter1D(x_list, y_list, x):
#
#     if not isinstance(x_list, torch.Tensor):
#         x_list = torch.tensor(x_list)
#     if not isinstance(y_list, torch.Tensor):
#         y_list = torch.tensor(y_list)
#     if not isinstance(x, torch.Tensor):
#         x = torch.tensor(x)
#     x_list = x_list.view(-1,1)
#     y_list = y_list.view(-1,1)
#     x = x.view(-1,1)
#     # x_list = torch.tensor(x_list).to(dtype=torch.float)
#     # x_list = x_list.unsqueeze(1).to(dtype=torch.float)
#     # y_list = y_list.unsqueeze(1).to(dtype=torch.float)
#     # x = x.view(-1,1).to(dtype=torch.float)
#     # 直接将x中的最大值与最小值作为x_list的最小、大边界
#
#     # x = torch.clamp(x, min=x_list[0,0], max=x_list[x_list.shape[0]-1,0])
#     x_min = torch.min(x)
#     x_max = torch.max(x)
#     # x = torch.clamp(x, min=x_list[0,0], max=x_list[x_list.shape[0]-1,0])
#
#     # 先尝试这个方法：
#     y_left = (y_list[1, 0] - y_list[0, 0]) / (x_list[1, 0] - x_list[0, 0]) * (x_min - x_list[0, 0]) + y_list[0, 0]
#     y_right = (y_list[-1, 0] - y_list[-2, 0]) / (x_list[-1, 0] - x_list[-2, 0]) * (x_max - x_list[-1, 0]) + y_list[-1, 0]
#
#     condition1 = (x_min<x_list[0,0]).repeat(2)
#     res1 = torch.cat((y_left.view(1), x_min.view(1)),dim=0)
#     res2 = torch.cat((y_list[0,0].view(1), x_list[0,0].view(1)),dim=0)
#     out1 = torch.where(condition1, res1, res2)
#     y_list[0,0] = out1[0]
#     x_list[0,0] = out1[1]
#
#     condition2 = (x_max>x_list[-1,0]).repeat(2)
#     res3 = torch.cat((y_right.view(1), x_max.view(1)),dim=0)
#     res4 = torch.cat((y_list[-1,0].view(1), x_list[-1,0].view(1)),dim=0)
#     out2 = torch.where(condition2, res3, res4)
#     y_list[-1,0] = out2[0]
#     x_list[-1,0] = out2[1]
#
#     # 如果上面的方法不ok，就用下面的方法：
#     # len_x = x_list.shape[0]
#     # len_y = y_list.shape[0]
#     # if x_min<x_list[0,0]:
#     #     y_left = (y_list[1,0] - y_list[0,0]) / (x_list[1,0] - x_list[0,0]) * (x_min - x_list[0,0]) + y_list[0,0]
#     #     y_list[0,0] = y_left
#     #     x_list[0,0] = x_min
#     # if x_max>x_list[-1,0]:
#     #     y_right = (y_list[len_y-1,0] - y_list[len_y-2,0]) / (x_list[len_x-1,0] - x_list[len_x-2,0]) * (x_max - x_list[len_x-1,0]) + y_list[len_y-1,0]
#     #     y_list[len_y-1,0] = y_right
#     #     x_list[len_x-1,0] = x_max
#
#     # 确保输入张量是连续的
#     # x = x.contiguous()
#
#     # 找到输入低压和高压在表中的位置
#     x_index = searchsorted(x_list, x)[:,0]
#
#     y = y_list[x_index] + (x - x_list[x_index]) * (y_list[x_index + 1] - y_list[x_index]) / (x_list[x_index + 1] - x_list[x_index])
#     return y

# 单线性插值
def inter1D(x_list, y_list, x):
    if not isinstance(x_list, torch.Tensor):
        x_list = torch.tensor(x_list)
    if not isinstance(y_list, torch.Tensor):
        y_list = torch.tensor(y_list)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    # x_list = x_list.unsqueeze(1).to(dtype=torch.float)
    # y_list = y_list.unsqueeze(1).to(dtype=torch.float)
    # x = x.view(-1,1).to(dtype=torch.float)

    # 直接将x中的最大值与最小值作为x_list的最小、大边界
    device = x.device
    x_list = x_list.view(-1, 1).to(device).contiguous()
    y_list = y_list.view(-1, 1).to(device).contiguous()
    x = x.view(-1, 1)

    # x = torch.clamp(x, min=x_list[0, 0], max=x_list[x_list.shape[0] - 1, 0])
    x_min = torch.min(x)
    x_max = torch.max(x)
    if x_min<x_list[0,0]:
        y_left = (y_list[1,0] - y_list[0,0]) / (x_list[1,0] - x_list[0,0]) * (x_min - x_list[0,0]) + y_list[0,0]
        y_list[0,0] = y_left
        x_list[0,0] = x_min
    if x_max>x_list[-1,0]:
        y_right = (y_list[-1,0] - y_list[-2,0]) / (x_list[-1,0] - x_list[-2,0]) * (x_max - x_list[-1,0]) + y_list[-1,0]
        y_list[-1,0] = y_right
        x_list[-1,0] = x_max

    # 确保输入张量是连续的
    # x = x.contiguous()

    # 找到输入低压和高压在表中的位置
    x_index = searchsorted(x_list, x)[:, 0]

    y = y_list[x_index] + (x - x_list[x_index]) * (y_list[x_index + 1] - y_list[x_index]) / (x_list[x_index + 1] - x_list[x_index])
    return y


# 双线性插值
# 输入x1_table, x2_table, y_table, x1, x2为b*1
# 输出y为b*1
def inter2D(x1_table, x2_table, y_table, x1, x2):  # 双线性插值
    with torch.no_grad():
        # 将输入的压力转换为 PyTorch 张量
        if isinstance(x1, list):
            x1 = torch.tensor(x1)
        if isinstance(x2, list):
            x2 = torch.tensor(x2)
        if isinstance(x2, list):
            x1_table = torch.tensor(x1_table)
        if isinstance(x2, list):
            x2_table = torch.tensor(x2_table)
        if isinstance(x2, list):
            y_table = torch.tensor(y_table)

        device = x1.device

        x1_table = x1_table.view(-1, 1).to(dtype=torch.float).to(device).contiguous()
        x2_table = x2_table.view(-1, 1).to(dtype=torch.float).to(device).contiguous()
        y_table = y_table.to(dtype=torch.float).to(device).contiguous()
        x1 = x1.view(-1, 1).to(dtype=torch.float)
        x2 = x2.view(-1, 1).to(dtype=torch.float)

        # 确保输入张量是连续的
        x1 = x1.contiguous()
        x2 = x2.contiguous()

        # x1 = torch.clamp(x1, min=x1_table[0, 0], max=x1_table[-1, 0])
        # 直接将表格插值补齐
        x1_min = torch.min(x1)
        x1_max = torch.max(x1)

        # y_table == x1_len * x2_len
        # 在x1左边添加一行，在y上面添加一行
        if x1_min < x1_table[0, 0]:
            y_left = (y_table[1, :] - y_table[0, :]) / (x1_table[1] - x1_table[0]) * (x1_min - x1_table[0]) + y_table[0, :]
            y_table[0, :] = y_left
            x1_table[0] = x1_min
        # 在x1右边添加一行，在y下面添加一行
        if x1_max > x1_table[-1, 0]:
            y_right = (y_table[-1, :] - y_table[-2, :]) / (x1_table[-1] - x1_table[-2]) * (x1_max - x1_table[-1]) + y_table[-1, :]
            y_table[-1, :] = y_right
            x1_table[-1] = x1_max

        # x2 = torch.clamp(x2, min=x2_table[0, 0], max=x2_table[-1, 0])
        x2_min = torch.min(x2)
        x2_max = torch.max(x2)

        # 在x2左边添加一行，在y左边添加一行
        if x2_min < x2_table[0, 0]:
            y_left = (y_table[:, 1] - y_table[:, 0]) / (x2_table[1] - x2_table[0]) * (x2_min - x2_table[0]) + y_table[:, 0]
            y_table[:, 0] = y_left
            x2_table[0] = x2_min
        # 在x2左边添加一行，在y右边添加一行
        if x2_max > x2_table[-1, 0]:
            y_left = (y_table[:, -1] - y_table[:, -2]) / (x2_table[-1] - x2_table[-2]) * (x2_max - x2_table[-1]) + y_table[:, -1]
            y_table[:, -1] = y_left
            x2_table[-1] = x2_max

        # 创建一个网格，用于查找

        # 找到输入低压和高压在表中的位置
        x1_idx = searchsorted(x1_table, x1)[:, 0]
        x2_idx = searchsorted(x2_table, x2)[:, 0]

        # 获取四个最近的点
        Q11 = y_table[x1_idx, x2_idx].view(-1, 1)
        Q12 = y_table[x1_idx, x2_idx + 1].view(-1, 1)
        Q21 = y_table[x1_idx + 1, x2_idx].view(-1, 1)
        Q22 = y_table[x1_idx + 1, x2_idx + 1].view(-1, 1)

        # 计算 x 和 y 方向的比例
        x_ratio = (x1 - torch.ones(x1.shape) * x1_table[x1_idx]) / (torch.ones(x1.shape) * (x1_table[x1_idx + 1] - x1_table[x1_idx]))

        y_ratio = (x2 - torch.ones(x1.shape) * x2_table[x2_idx]) / (torch.ones(x1.shape) * (x2_table[x2_idx + 1] - x2_table[x2_idx]))

        # 在 x 方向上进行线性插值
        R1 = torch.mul(x_ratio, (Q21 - Q11)) + Q11
        R2 = torch.mul(x_ratio, (Q22 - Q12)) + Q12

        # 在 y 方向上进行线性插值
        P = torch.mul(y_ratio, (R2 - R1)) + R1
    return P


def searchsorted(sorted_sequence, values, out_int32: bool = False) -> torch.LongTensor:
    """
    手动实现 torch.searchsorted 功能。

    参数:
        sorted_sequence (Tensor): 一个有序的一维张量。
        values (Tensor or Scalar): 要查找插入位置的值。
        out_int32 (bool, optional): 输出索引的类型是否为 int32，默认为 False（int64）。

    返回:
        Tensor: 插入位置左边的索引，维度为b*1。
    """
    # values = values.view(-1,1)
    # indices = torch.zeros_like(values)
    #
    # for i in range(sorted_sequence.shape[0]-1):
    #     condition = (sorted_sequence[i,:] <= values) & (values <= sorted_sequence[i+1,:]) & (indices == torch.zeros_like(values))
    #     # res = torch.where(condition, torch.full_like(values,i), torch.zeros_like(values))
    #     # indices = indices + res
    #     indices[condition] = i

    sorted_sequence = sorted_sequence.view(-1, 1)
    # 初始化 indices
    indices = torch.zeros_like(values, dtype=torch.long)

    # 计算条件矩阵
    conditions = (sorted_sequence.unsqueeze(1) <= values) & (
            values <= sorted_sequence.unsqueeze(1).roll(-1, dims=0)) & (indices == torch.zeros_like(values))

    # 计算索引矩阵
    # 更新 indices
    indices = torch.argmax(conditions[:, :, 0].detach().to(dtype=torch.int), dim=0)

    indices = torch.clamp(indices, 0, sorted_sequence.shape[0] - 2)
    indices = indices.to(torch.int64).view(-1, 1)
    return indices



# air_temp_after_heat_exchange
def cal_air_temp_after_heat_exchange(tem_df, air_temp_before_heat_exchange, mix_door_pwm_dict, all_vol_rate_channel):
    # 出风温度
    all_wind_channel_before_mix_door_dict = {}
    for mix_door_pwm_name in config_common.mix_door_control:
        mix_door_pwm = mix_door_pwm_dict[mix_door_pwm_name]
        all_controlled_wind_channel_temp_name = config_common.mix_door_control[mix_door_pwm_name]
        for name_index in range(len(all_controlled_wind_channel_temp_name)):
            controlled_wind_channel_temp_name = all_controlled_wind_channel_temp_name[name_index]
            wind_channel_temp = np.array(tem_df[controlled_wind_channel_temp_name])

            not_use_before = mix_door_pwm != 0
            wind_channel_temp_before_mix_door = np.array(air_temp_before_heat_exchange)
            wind_channel_temp_before_mix_door[not_use_before] = (wind_channel_temp[not_use_before] -
                                                                 air_temp_before_heat_exchange[not_use_before] * (1 - mix_door_pwm[not_use_before])) / \
                                                                (mix_door_pwm[not_use_before] + 1e-9)
            all_wind_channel_before_mix_door_dict[f'{controlled_wind_channel_temp_name}_before'] = wind_channel_temp_before_mix_door

    all_wind_channel_temp = []
    for key_index in range(len(config_common.all_wind_channel_temp_keys)):
        wind_temp_key = config_common.all_wind_channel_temp_keys[key_index]
        wind_temp = all_wind_channel_before_mix_door_dict[f'{wind_temp_key}_before']
        all_wind_channel_temp.append(wind_temp)

    all_wind_channel_temp = np.array(all_wind_channel_temp).T

    # 排除无效值
    air_temp_after_heat_exchange = []
    for time_index in range(len(all_wind_channel_temp)):
        now_all_wind_channel_temp = all_wind_channel_temp[time_index]
        ok_mask = now_all_wind_channel_temp < config_common.max_channel_temp
        used_all_vol_rate_channel = all_vol_rate_channel[time_index][ok_mask]
        used_wind_channel_temp = now_all_wind_channel_temp[ok_mask]
        now_wind_temp_weight = used_all_vol_rate_channel / (np.sum(used_all_vol_rate_channel, axis=-1, keepdims=True) + 1e-10)
        now_air_temp_after_heat_exchange = np.sum(used_wind_channel_temp * now_wind_temp_weight, axis=-1)
        air_temp_after_heat_exchange.append(now_air_temp_after_heat_exchange)
    air_temp_after_heat_exchange = np.array(air_temp_after_heat_exchange)
    return air_temp_after_heat_exchange


# out_frac
def cal_out_frac(tem_df, car_type):
    # 外循环比例
    inlet_pwm = np.array(tem_df['Inlet_PWM'])
    if car_type == 'modena':
        out_frac = interp_from_table(inlet_pwm, config_common.inlet_pwm_to_out_frac_modena[:, 0],
                                     config_common.inlet_pwm_to_out_frac_modena[:, 1]) / 100

        inlet_gear = interp_from_table(out_frac * 100, config_common.inlet_gear_to_out_frac_modena[:, 1],
                                       config_common.inlet_gear_to_out_frac_modena[:, 0])
    elif 'lemans' in car_type:
        inlet_pwm_d_lin = np.array(tem_df['Inlet_PWM_D_LIN'])
        use_dc_place = inlet_pwm < 100
        out_frac = np.zeros_like(inlet_pwm_d_lin)
        if use_dc_place.any():
            use_dc_place_out_frac = interp_from_table(inlet_pwm[use_dc_place], config_common.inlet_pwm_to_out_frac_lemans[:, 0],
                                                      config_common.inlet_pwm_to_out_frac_lemans[:, 1]) / 100
            out_frac[use_dc_place] = use_dc_place_out_frac
        if (~use_dc_place).any():
            use_lin_place_out_frac = interp_from_table(inlet_pwm_d_lin[~use_dc_place], config_common.inlet_pwm_d_lin_to_out_frac_lemans[:, 0],
                                                       config_common.inlet_pwm_d_lin_to_out_frac_lemans[:, 1]) / 100

            out_frac[~use_dc_place] = use_lin_place_out_frac

        inlet_gear = interp_from_table(out_frac * 100, config_common.inlet_gear_to_out_frac_lemans[:, 1],
                                       config_common.inlet_gear_to_out_frac_lemans[:, 0])

    # tem_df['out_frac'] = out_frac
    return out_frac, inlet_gear


# 限制模型参数范围 包括最大最小 保留位数
def clip_wights(net, max_data=65504, min_data=-65504, decimals=4):
    for name, param in net.named_parameters():
        with torch.no_grad():
            # 保留小数
            param.data = torch.round(param.data, decimals=decimals)

            # 裁剪参数
            param.data = torch.clamp(param.data, min_data, max_data)


# 计算阻力功率
def cal_w_pwr(ags_openness, car_speed, car_type):
    V = car_speed / 3.6
    if car_type == 'modena':
        C_d = torch.relu((-0.00444 * ags_openness ** 2 + 0.7467 * ags_openness - 0.1) * 0.001)
        w_pwr = 0.5 * config_common.W_p * config_common.W_A_modena * C_d * V ** 3
    elif 'lemans' in car_type:
        C_d = torch.relu(-2.69e-6 * ags_openness ** 2 + 4.86e-4 * ags_openness - 3.35e-3)
        w_pwr = 0.5 * config_common.W_p * config_common.W_A_lemans * C_d * V ** 3

    return w_pwr


# 计算气体密度
def get_air_density(temp):
    air_density = config_common.zero_air_density * (-config_common.zero_temp_bias) / ((-config_common.zero_temp_bias) + temp)
    return air_density


def get_device(device_id) -> torch.device:
    return torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")


# 插值
def interp_from_table(data, table_key, table_value):
    # 创建插值函数
    interp_func = interp1d(table_key, table_value, kind='linear')

    # 进行插值
    interpolated_data = interp_func(data)

    return interpolated_data


# 插值 2D
def interp_from_table_2d(key_1, key_2, table_key_1, table_key_2, table_value):
    # 创建插值函数
    # interp_func = interp2d(table_key_1, table_key_2, table_value, kind='linear')
    interp_func = interp2d(table_key_2, table_key_1, table_value, kind='linear')
    # 进行插值
    # interpolated_data = interp_func(key_1, key_2)

    data = []
    for i in range(len(key_1)):
        # data.append(interp_func(key_1[i], key_2[i]))
        data.append(interp_func(key_2[i], key_1[i]))

    return data
