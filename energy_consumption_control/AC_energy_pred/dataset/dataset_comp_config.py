import os
from tqdm import tqdm
import numpy as np
import torch

import config_all as config
from data_utils import read_csv


# 最大最小归一化
def norm_data(data, max_data, min_data):
    normed_data = (data - min_data) / (max_data - min_data)
    return normed_data


def recover_data(normed_data, max_data, min_data):
    data = normed_data * (max_data - min_data) + min_data
    return data


class CompConfigDataset(torch.utils.data.Dataset):
    def __init__(self, all_data_path, return_point):
        self.all_data_path = all_data_path
        self.return_point = return_point

        self.all_x = []
        self.all_y = []
        self.all_info = []

        for path_index in tqdm(range(len(self.all_data_path))):
            data_path = self.all_data_path[path_index]
            out_dict = read_csv(data_path)

            # 输入参数
            hi_pressure = out_dict['hi_pressure']
            temp_p_h_2 = out_dict['temp_p_h_2']
            warmer_p = out_dict['warmer_p']
            wind_side_cool_medium_temp_out = out_dict['wind_side_cool_medium_temp_out']
            wind_side_cool_medium_temp_in = out_dict['wind_side_cool_medium_temp_in']
            pump_speed = out_dict['pump_speed']
            dwt = out_dict['dwt']

            x = [hi_pressure] + [temp_p_h_2] + [warmer_p] + [pump_speed] + [wind_side_cool_medium_temp_out] + [wind_side_cool_medium_temp_in] + [dwt]
            x = np.array(x).T

            # 开始到最后前一位
            x = x[:-1]

            # 输出参数
            ac_hp_tar = out_dict['aim_hi_pressure']
            # ac_hp_tar_next = ac_hp_tar[1:]
            # ac_hp_tar_next[-1] = ac_hp_tar[-1]
            # ac_hp_tar_next = np.log(ac_hp_tar_next)  # 应用对数变换

            y = [ac_hp_tar]
            y = np.array(y).T

            # 第二到最后一位
            y = y[1:]

            # 过滤条件
            # wind_vol = out_dict['wind_vol']
            hp_mode = out_dict['hp_mode']
            # cab_cooling_status_act_pos = out_dict['cab_cooling_status_act_pos']

            # mask = (wind_vol <= 50) | (cab_cooling_status_act_pos != 0) | (hp_mode != config.heat_mode_index)
            mask = (hp_mode != config.heat_mode_index)
            mask_x = np.ma.array(x, mask=np.repeat(mask.reshape(-1, 1), x.shape[-1], axis=-1))
            clumps = np.ma.clump_unmasked(mask_x[:, 0])

            all_split_x = []
            all_split_y = []
            for split_index in range(len(clumps)):
                split_range = clumps[split_index]

                split_x = x[split_range]
                split_y = y[split_range]

                if len(split_x) < config.min_split_len:
                    print(f'len(split_x) < {config.min_split_len}')
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
