import os
from tqdm import tqdm
import numpy as np
import torch

import config_all
from data_utils import read_csv


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
class HeatingActorBaseDataset(torch.utils.data.Dataset):

    def __init__(self, all_data_path, return_point):
        self.all_data_path = all_data_path
        self.return_point = return_point

        self.all_x = []
        self.all_y = []
        self.all_info = []

        for path_index in tqdm(range(len(self.all_data_path))):
            data_path = self.all_data_path[path_index]
            out_dict = read_csv(data_path)

            lo_pressure_temp = out_dict['lo_pressure_temp']
            hi_pressure_temp = out_dict['hi_pressure_temp']

            cab_heating_status_act_pos = out_dict['cab_heating_status_act_pos']
            compressor_speed = out_dict['compressor_speed']

            temp_p_h_1 = out_dict['temp_p_h_1_cab_heating']
            temp_p_h_2 = out_dict['temp_p_h_2']
            temp_p_h_5 = out_dict['temp_p_h_5']

            car_speed = out_dict['car_speed']
            temp_amb = np.array(out_dict['temp_amb'])

            hp_mode = out_dict['hp_mode']

            ags_openness=out_dict['ags_openness']
            cfan_pwm = out_dict['cfan_pwm']

            x = [lo_pressure_temp] + [hi_pressure_temp] + [cab_heating_status_act_pos] + [compressor_speed] + [temp_p_h_1] + [temp_p_h_2] + \
                [temp_p_h_5] + [car_speed] + [temp_amb]
            x = np.array(x).T

            # 转化为分类label
            ags_openness_step=15
            cfan_pwm_step=1
            ags_openness_index =ags_openness//ags_openness_step
            cfan_pwm_index  = cfan_pwm//cfan_pwm_step
            y = [ags_openness_index] + [cfan_pwm_index]
            y = np.array(y).T

            # 模式10 或者开了加热器的
            mask = (hp_mode != config_all.heat_mode_index)
            mask_x = np.ma.array(x, mask=np.repeat(mask.reshape(-1, 1), x.shape[-1], axis=-1))
            # mask_y = ma.array(y, mask=np.repeat(mask.reshape(-1,1),y.shape[-1],axis=-1))
            clumps = np.ma.clump_unmasked(mask_x[:, 0])

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
