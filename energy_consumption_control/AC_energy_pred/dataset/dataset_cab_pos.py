import os
from tqdm import tqdm
import numpy as np
import torch

from AC_energy_pred import config_all
from AC_energy_pred.data_utils import read_csv


# 最大最小归一化
def norm_data(data, max_data, min_data):
    normed_data = (data - min_data) / (max_data - min_data)
    return normed_data


def recover_data(normed_data, max_data, min_data):
    data = normed_data * (max_data - min_data) + min_data
    return data


# 根据开度估算制冷剂流量 # todo 未完成待定
def cal_refrigerant_vol_para(battery_cooling_status_act_pos, cab_cooling_status_act_pos, cab_heating_status_act_pos, hp_mode, compressor_speed):
    refrigerant_vol_para = np.zeros_like(hp_mode)
    refrigerant_vol_para[hp_mode == config_all.heat_mode_index] = cab_heating_status_act_pos[hp_mode == config_all.heat_mode_index]
    refrigerant_vol_para[hp_mode == config_all.cooling_mode_index] = cab_cooling_status_act_pos[hp_mode == config_all.cooling_mode_index]
    return refrigerant_vol_para


# 制热模式数据 使用数据规则得到的ags开度风扇占空比 作为学习目标
class CabPosBaseDataset(torch.utils.data.Dataset):

    def __init__(self, all_data_path, return_point):
        self.all_data_path = all_data_path
        self.return_point = return_point

        self.all_x = []
        self.all_y = []
        self.all_info = []

        for path_index in tqdm(range(len(self.all_data_path))):
            data_path = self.all_data_path[path_index]
            out_dict = read_csv(data_path)

            # 目标过冷度
            sc_tar_mode_10 = out_dict['sc_tar_mode_10'][1:]
            # 目标过热度
            sh_tar_mode_10 = out_dict['sh_tar_mode_10'][1:]
            # 压缩机进气温度
            temp_p_h_1_cab_heating = out_dict['temp_p_h_1_cab_heating'][:-1]
            # 内冷温度
            temp_p_h_5 = out_dict['temp_p_h_5'][:-1]

            # 上一时刻饱和高压
            hi_pressure = out_dict['hi_pressure'][:-1]
            # 上一时刻饱和低压
            lo_pressure = out_dict['lo_pressure'][:-1]

            # 上一时刻压缩机转速
            compressor_speed_last = out_dict['compressor_speed'][:-1]

            # 输出
            # 膨胀阀开度
            cab_heating_status_act_pos = out_dict['cab_heating_status_act_pos'][1:]

            # 过滤条件
            # 只考虑模式10
            hp_mode = out_dict['hp_mode'][1:]
            # 不考虑过热度为负的情况
            sh_act_mode_10 = out_dict['sh_act_mode_10'][1:]

            # x = [ac_pid_out_hp] + [lo_pressure] + [hi_pressure] + [aim_lo_pressure] + [aim_hi_pressure] + \
            #     [exv_pid_iout] +[sc_tar_mode_10] + [sh_tar_mode_10] + \
            #     [temp_p_h_2] + [temp_p_h_5] + [temp_p_h_1_cab_heating]
            x = [sc_tar_mode_10] + [sh_tar_mode_10] + [temp_p_h_1_cab_heating] + [temp_p_h_5] + [hi_pressure] + [lo_pressure] + [compressor_speed_last]
            x = np.array(x).T
            # mask = (wind_vol <= 50) | ((hp_mode != config_all.heat_mode_index)) | (
            #         warmer_p > 0) | (cab_cooling_status_act_pos != 0)
            # mask = (hp_mode != config_all.heat_mode_index) | (aim_hi_pressure > aim_lo_pressure) | (temp_p_h_2 > temp_p_h_1_cab_heating)
            mask = (hp_mode != config_all.heat_mode_index)| (sh_act_mode_10 < 0) | (cab_heating_status_act_pos < 12)

            print(f"x shape:{x.shape}")
            print(f"mask shape:{mask.shape}")
            mask_x = np.ma.array(x, mask=np.repeat(mask.reshape(-1, 1), x.shape[-1], axis=-1))
            # mask_y = ma.array(y, mask=np.repeat(mask.reshape(-1,1),y.shape[-1],axis=-1))
            clumps = np.ma.clump_unmasked(mask_x[:, 0])

            y = [cab_heating_status_act_pos]
            y = np.array(y).T

            all_split_x = []
            all_split_y = []
            for split_index in range(len(clumps)):
                split_range = clumps[split_index]

                split_x = x[split_range]
                split_y = y[split_range]

                if len(split_x) < config_all.min_split_len:
                    continue

                all_split_x.append(split_x.astype('float32'))
                all_split_y.append(split_y.astype('float32'))

            if True in np.isnan(x) or True in np.isnan(y):
                print(path_index, data_path, 'error')
                continue

            if len(all_split_x) == 0:
                print(path_index, data_path, 'no ok split')
                continue

            if return_point:
                self.all_y.append(np.concatenate(all_split_y))
                self.all_x.append(np.concatenate(all_split_x))
            else:
                self.all_y.append(all_split_y)
                self.all_x.append(all_split_x)
                self.all_info.append([f'{data_path}_{i}' for i in range(len(clumps))])

        if return_point:
            self.all_x = np.concatenate(self.all_x)
            self.all_y = np.concatenate(self.all_y)
            # self.all_info = np.concatenate(self.all_info)

    def __len__(self):
        return len(self.all_x)

    def __getitem__(self, item_index):

        x = self.all_x[item_index]
        y = self.all_y[item_index]

        return x, y
