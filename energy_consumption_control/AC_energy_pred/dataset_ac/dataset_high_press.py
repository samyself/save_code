import os
from tqdm import tqdm
import numpy as np
import torch

from AC_energy_pred import config_all
from AC_energy_pred.data_utils import read_csv
# import os
# from tqdm import tqdm
# import numpy as np
# import torch
#
# import config_common
# from AC_energy_pred import config_ac
# from AC_energy_pred.data_utils_ac import read_csv


# 制热模式数据 使用数据规则得到的ags开度风扇占空比 作为学习目标
class HighPressBaseDataset(torch.utils.data.Dataset):

    def __init__(self, all_data_path, return_point):
        self.all_data_path = all_data_path
        self.return_point = return_point

        self.all_x = []
        self.all_y = []
        self.all_info = []

        for path_index in tqdm(range(len(self.all_data_path))):
            data_path = self.all_data_path[path_index]
            # if 'LT031' in data_path:
            #     out_dict = read_csv(data_path,car_type='lemans_low')
            # elif 'LV222' in data_path or 'lemans-2' in data_path:
            #     out_dict = read_csv(data_path,car_type='lemans_mid')
            # else:
            #     out_dict = read_csv(data_path,car_type='lemans_adv')
            out_dict = read_csv(data_path, car_type='lemans_adv')
            '''当前值'''
            # 蒸发器的风温
            air_temp_before_heat_exchange = out_dict['air_temp_before_heat_exchange'][1:]
            # 蒸发器的风量
            wind_vol = out_dict['wind_vol'][1:]
            # 压缩机转速
            compressor_speed = out_dict['compressor_speed'][1:]
            last_compressor_speed = out_dict['compressor_speed'][:-1]
            # compressor_speed_diff = compressor_speed - last_compressor_speed

            # 当前时刻目标饱和高压
            aim_hi_pressure = out_dict['aim_hi_pressure'][1:]
            # 当前时刻目标饱和低压
            aim_lo_pressure = out_dict['aim_lo_pressure'][1:]
            # CEXV膨胀阀开度
            cab_heating_status_act_pos = out_dict['cab_heating_status_act_pos'][1:]
            # 当前制热水泵流量
            heat_coolant_vol = out_dict['heat_coolant_vol'][1:]
            # 当前换热前的水温 todo 暂无数据
            cool_medium_temp_in = np.zeros_like(heat_coolant_vol)
            # hvch出水温度
            hvch_cool_medium_temp_out = out_dict['hvch_cool_medium_temp_out'][1:]
            '''上一时刻值'''
            # 压缩机进气温度
            temp_p_h_1_cab_heating = out_dict['temp_p_h_1_cab_heating'][:-1]
            # 饱和低压
            lo_pressure = out_dict['lo_pressure'][:-1]
            # 内冷温度
            temp_p_h_5 = out_dict['temp_p_h_5'][:-1]

            # 过滤条件
            # wind_vol = out_dict['wind_vol'][1:]
            hp_mode = out_dict['hp_mode'][1:]
            # cab_cooling_status_act_pos = out_dict['cab_cooling_status_act_pos'][1:]
            # warmer_p = out_dict['warmer_p'][1:]

            x = [air_temp_before_heat_exchange] + [wind_vol] + [compressor_speed] + [cab_heating_status_act_pos] + [temp_p_h_1_cab_heating] + \
                [lo_pressure] + [temp_p_h_5] + [heat_coolant_vol] + [cool_medium_temp_in] + [hvch_cool_medium_temp_out] \
                + [aim_hi_pressure]  + [last_compressor_speed] + [aim_lo_pressure]
            x = np.array(x).T
            # mask = (wind_vol <= 50) | (hp_mode != config_ac.heat_mode_index) | (
            #         warmer_p > 0) | (cab_cooling_status_act_pos != 0)
            mask = hp_mode != config_all.heat_mode_index
            print(f"x shape:{x.shape}")
            print(f"mask shape:{mask.shape}")
            mask_x = np.ma.array(x, mask=np.repeat(mask.reshape(-1, 1), x.shape[-1], axis=-1))
            # mask_y = ma.array(y, mask=np.repeat(mask.reshape(-1,1),y.shape[-1],axis=-1))
            clumps = np.ma.clump_unmasked(mask_x[:, 0])

            # 输出
            # 下一时刻内冷温度
            temp_p_h_5 = out_dict['temp_p_h_5'][1:]
            # 压缩机排气温度
            temp_p_h_2 = out_dict['temp_p_h_2'][1:]
            # 饱和高压
            hi_pressure = out_dict['hi_pressure'][1:]

            y = [temp_p_h_5] + [temp_p_h_2] + [hi_pressure]
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
