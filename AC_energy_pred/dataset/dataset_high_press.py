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
class HighPressBaseDataset(torch.utils.data.Dataset):

    def __init__(self, all_data_path, return_point):
        self.all_data_path = all_data_path
        self.return_point = return_point

        self.all_x = []
        self.all_y = []
        self.all_info = []

        for path_index in tqdm(range(len(self.all_data_path))):
            data_path = self.all_data_path[path_index]
            out_dict = read_csv(data_path)
            if len(out_dict['air_temp_before_heat_exchange']) >= 181 + config_all.min_split_len:
                pass
            else:
                continue
            '''当前值'''
            # 蒸发器的风温
            air_temp_before_heat_exchange = out_dict['air_temp_before_heat_exchange'][1+180:]
            # 蒸发器的风量
            wind_vol = out_dict['wind_vol'][1+180:]
            # 压缩机转速
            compressor_speed = out_dict['compressor_speed'][1+180:]
            # 压缩机速率
            compressor_speed_change = np.abs(out_dict['compressor_speed'][1+180:] - out_dict['compressor_speed'][180:-1])
            # CEXV膨胀阀开度
            cab_heating_status_act_pos = out_dict['cab_heating_status_act_pos'][1+180:]
            # 当前制热水泵流量
            heat_coolant_vol = out_dict['heat_coolant_vol'][1+180:]
            # 当前换热前的水温 todo 暂无数据
            cool_medium_temp_in = np.zeros_like(heat_coolant_vol)
            # hvch出水温度
            hvch_cool_medium_temp_out = out_dict['hvch_cool_medium_temp_out'][1+180:]
            '''上一时刻值'''
            # 压缩机进气温度
            temp_p_h_1_cab_heating = out_dict['temp_p_h_1_cab_heating'][180:-1]
            # 饱和低压
            lo_pressure = out_dict['lo_pressure'][180:-1]
            # 上一时刻饱和高压
            last_hi_pressure = out_dict['hi_pressure'][180:-1]

            # 内冷温度
            temp_p_h_5 = out_dict['temp_p_h_5'][180:-1]
            # 压比
            pressure_ratio = out_dict['hi_pressure'][1+180:] / lo_pressure
            # 过滤条件
            # wind_vol = out_dict['wind_vol'][1:]
            hp_mode = out_dict['hp_mode'][1+180:]
            # cab_cooling_status_act_pos = out_dict['cab_cooling_status_act_pos'][1:]
            # warmer_p = out_dict['warmer_p'][1:]

            # 输出
            # 下一时刻内冷温度
            temp_p_h_5 = out_dict['temp_p_h_5'][1+180:]
            # 压缩机排气温度
            temp_p_h_2 = out_dict['temp_p_h_2'][1+180:]
            # 饱和高压
            hi_pressure = out_dict['hi_pressure'][1+180:]

            x = [air_temp_before_heat_exchange] + [wind_vol] + [compressor_speed] + [cab_heating_status_act_pos] + [temp_p_h_1_cab_heating] + \
                [lo_pressure] + [temp_p_h_5] + [heat_coolant_vol] + [cool_medium_temp_in] + [hvch_cool_medium_temp_out]
            x = np.array(x).T
            # mask = (wind_vol <= 50) | (hp_mode != config_all.heat_mode_index) | (
            #         warmer_p > 0) | (cab_cooling_status_act_pos != 0)
            # mask = (hp_mode != config_all.heat_mode_index) | (compressor_speed_change > 1150) | (compressor_speed < 800) | (
            #         compressor_speed > 8000) | (temp_p_h_2 < 40) | (temp_p_h_2 > 110)
            mask = (hp_mode != config_all.heat_mode_index)
            print(f"x shape:{x.shape}")
            print(f"mask shape:{mask.shape}")
            mask_x = np.ma.array(x, mask=np.repeat(mask.reshape(-1, 1), x.shape[-1], axis=-1))
            # mask_y = ma.array(y, mask=np.repeat(mask.reshape(-1,1),y.shape[-1],axis=-1))
            clumps = np.ma.clump_unmasked(mask_x[:, 0])

            y = [temp_p_h_5] + [temp_p_h_2] +  [hi_pressure]
            y = np.array(y).T
            all_split_x = []
            all_split_y = []
            for split_index in range(len(clumps)):
                split_range = clumps[split_index]

                split_x = x[split_range]
                split_y = y[split_range]

                if len(split_x) < config_all.min_split_len:
                    continue
                # #-----取时序窗口-------------------------------
                # _x = []
                # _y = []
                # for i in range(len(split_x)-30):
                #     _x.append(split_x[i:i+30])
                #     _y.append(split_y[i:i+30])
                # split_x = np.array(_x).reshape(-1, 30, 10)
                # split_y = np.array(_y).reshape(-1, 30, 3)



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
