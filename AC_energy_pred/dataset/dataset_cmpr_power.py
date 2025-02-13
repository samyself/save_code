import math
import os
from tqdm import tqdm
import numpy as np
import torch

import config
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
    refrigerant_vol_para[hp_mode == config.heat_mode_index] = cab_heating_status_act_pos[hp_mode == config.heat_mode_index]
    refrigerant_vol_para[hp_mode == config.cooling_mode_index] = cab_cooling_status_act_pos[hp_mode == config.cooling_mode_index]
    return refrigerant_vol_para


class CmprPowerDatasetHeating(torch.utils.data.Dataset):
    compressor_p_max = 8000
    compressor_p_min = 0

    def __init__(self, all_data_path, return_point):
        self.all_data_path = all_data_path
        self.return_point = return_point

        self.all_x = []
        self.all_y = []
        self.all_info = []

        for path_index in tqdm(range(len(self.all_data_path))):
            data_path = self.all_data_path[path_index]
            out_dict = read_csv(data_path)

            # wind_vol = out_dict['wind_vol']
            # hp_mode = out_dict['hp_mode']

            # x = [air_temp_before_heat_exchange] + [wind_vol] + [ac_heat_exchange_temp] + [refrigerant_vol_para] + \
            #     [battery_cooling_status_act_pos] + [cab_cooling_status_act_pos] + [cab_heating_status_act_pos] + [hp_mode] + [compressor_speed]

            # x = [air_temp_before_heat_exchange] + [wind_vol] + [refrigerant_mix_temp] + [temp_p_h_2] + [temp_p_h_5] + [cab_heating_status_act_pos]

            # 压缩机转速
            TmRteOut_CmprSpdReq = out_dict['compressor_speed']
            # 压缩机排量
            Cmpr_capacity = out_dict['cmpr_capacity']
            # 饱和高压
            TmRteOut_HisideP = out_dict['hi_pressure']
            # 压缩机排气温度
            TmRteOut_CmprDchaT = out_dict['temp_p_h_2']
            # 压缩机进气温度：
            TmRteOut_OutrCondOutlT = out_dict['temp_p_h_1_cab_heating']
            # 饱和低压：
            TmRteOut_LosideP = out_dict['lo_pressure']
            # 压缩机功率
            compressor_p = out_dict['compressor_p']
            TmRteOut_CmprPwrCnsAct = norm_data(compressor_p, self.compressor_p_max, self.compressor_p_min)

            # TmRteOut_CmprPwrCnsAct = (out_dict['compressor_p'] - 0) / 8000.0
            x = [TmRteOut_CmprSpdReq] + [Cmpr_capacity] + [TmRteOut_HisideP] + [TmRteOut_CmprDchaT] + [TmRteOut_OutrCondOutlT] + [TmRteOut_LosideP]
            # x = [TmRteOut_CmprSpdReq]
            # x = np.array(x)
            x = np.array(x).T

            # warmer_p = out_dict['warmer_p']
            # cab_cooling_status_act_pos = out_dict['cab_cooling_status_act_pos']
            # 排除风量为0或电流为0的数据 模式不是2或10 或者开了加热器的
            # mask = (wind_vol <= 50) | ((hp_mode != config.cooling_mode_index) & (hp_mode != config.heat_mode_index)) | (warmer_p > 0) | (cab_cooling_status_act_pos != 0)
            # mask = (wind_vol <= 50) | ((hp_mode != config.heat_mode_index)) | (
            #             warmer_p > 0) | (cab_cooling_status_act_pos != 0)

            # 无需其他条件 没有不要的数据
            mask = np.zeros(len(x), dtype=bool)

            print(f"x shape:{x.shape}")
            # print(f"mask shape:{mask.shape}")
            mask_x = np.ma.array(x, mask=np.repeat(mask.reshape(-1, 1), x.shape[-1], axis=-1))
            # mask_y = ma.array(y, mask=np.repeat(mask.reshape(-1,1),y.shape[-1],axis=-1))
            clumps = np.ma.clump_unmasked(mask_x[:, 0])

            # 压缩机功率
            y = [TmRteOut_CmprPwrCnsAct]
            y = np.array(y).T

            all_split_x = []
            all_split_y = []
            for split_index in range(len(clumps)):
                split_range = clumps[split_index]

                split_x = x[split_range]
                split_y = y[split_range]

                if len(split_x) < config.min_split_len:
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
