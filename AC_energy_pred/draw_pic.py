import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import config

from data_utils import interp_from_table, read_csv, get_air_density


# 绘图
def draw_pred(series_list, series_name_list, pic_folder, pic_name):
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    total_sub_pic_num = len(series_list)
    plt.figure(figsize=(15, total_sub_pic_num * 6))

    for pic_index in range(total_sub_pic_num):
        sub_pic_data = series_list[pic_index]
        sub_pic_name = series_name_list[pic_index]
        plt.subplot(total_sub_pic_num, 1, pic_index + 1)
        for sub_index in range(len(sub_pic_name)):
            data = sub_pic_data[sub_index]
            name = sub_pic_name[sub_index]
            x = np.arange(len(data))
            plt.plot(x, data, label=name)
            plt.legend()

    if not os.path.exists(pic_folder):
        os.makedirs(pic_folder)
    pic_path = os.path.join(pic_folder, pic_name)

    plt.tight_layout(pad=1.08)
    plt.savefig(pic_path)
    plt.close()


if __name__ == '__main__':
    data_folder = './data/csv'
    all_file_name = os.listdir(data_folder)
    pic_folder = './data/pic'
    if not os.path.exists(pic_folder):
        os.makedirs(pic_folder)

    for file_index in tqdm(range(len(all_file_name))):
        file_name = all_file_name[file_index]
        data_path = os.path.join(data_folder, file_name)
        out_dict = read_csv(data_path)

        # 换热量
        air_heat_change = (out_dict['air_temp_after_heat_exchange'] - out_dict['air_temp_before_heat_exchange']) * out_dict['wind_vol']

        # 高压温度预估值
        esm_heat_change = out_dict['cab_heating_status_act_pos'] * out_dict['hi_pressure_temp']

        series_list = [
            [out_dict['resistance_p'], out_dict['compressor_p'], out_dict['blower_p'], out_dict['fan_p']],
            [out_dict['all_p']],
            [out_dict['warmer_p']],
            [out_dict['car_speed']],
            [out_dict['hi_pressure'], out_dict['lo_pressure']],

            [out_dict['hp_mode']],
            [out_dict['hvac_mode']],
            [out_dict['lo_pressure_temp'], out_dict['hi_pressure_temp']],
            # [out_dict['lo_pressure_temp'], out_dict['hi_pressure_temp'], out_dict['temp_p_h_2'], out_dict['temp_p_h_5'], out_dict['air_temp_after_heat_exchange']],
            [out_dict['lo_pressure_temp'], out_dict['hi_pressure_temp'], out_dict['temp_p_h_1_cab_cooling'],
             out_dict['air_temp_after_heat_exchange'], out_dict['air_temp_before_heat_exchange']],

            [out_dict['air_temp_before_heat_exchange'], out_dict['air_temp_after_heat_exchange'],
             out_dict['wind_side_cool_medium_temp_out'], out_dict['wind_side_cool_medium_temp_in']],
            [out_dict['wind_vol']],
            [out_dict['h_p_h_5'], out_dict['h_p_h_1_cab_cooling']],
            [out_dict['p_h_6_g_l_rate']],
            [out_dict['wind_heat_exchange']],
            [out_dict['cab_refrigerant_vol']],
            [out_dict['cab_heating_status_act_pos'], out_dict['cab_cooling_status_act_pos'], out_dict['battery_cooling_status_act_pos']],
            [out_dict['compressor_speed']],
        ]

        series_name_list = [
            ['resistance_p', 'compressor_p', 'blower_p', 'fan_p'],
            ['all_p'],
            ['warmer_p'],
            ['car_speed'],
            ['hi_pressure', 'lo_pressure'],

            ['hp_mode'],
            ['hvac_mode'],
            ['lo_pressure_temp', 'hi_pressure_temp'],
            # ['lo_pressure_temp', 'hi_pressure_temp', 'temp_p_h_2', 'temp_p_h_5', 'air_temp_after_heat_exchange',],
            ['lo_pressure_temp', 'hi_pressure_temp', 'temp_p_h_1_cab_cooling',
             'air_temp_after_heat_exchange', 'air_temp_before_heat_exchange'],

            ['air_temp_before_heat_exchange', 'air_temp_after_heat_exchange', 'wind_side_cool_medium_temp_out', 'wind_side_cool_medium_temp_in'],
            ['wind_vol'],
            ['h_p_h_5', 'h_p_h_1_cab_cooling'],
            ['p_h_6_g_l_rate'],
            ['wind_heat_exchange'],
            ['cab_refrigerant_vol'],
            ['cab_heating_status_act_pos', 'cab_cooling_status_act_pos', 'battery_cooling_status_act_pos'],
            ['compressor_speed'],
        ]

        pic_name = file_name.replace('.csv', '.png')
        draw_pred(series_list, series_name_list, pic_folder, pic_name)
