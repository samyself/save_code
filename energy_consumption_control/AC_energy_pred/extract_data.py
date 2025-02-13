import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import signal
import os
from pathlib import Path
import random
from asammdf import MDF
from asammdf.mdf import MdfException
import json
import re
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config_all

from data_utils import *

'''
提取业务方提供的原始能耗数据（所有工况）
将mdf格式数据转化为npy数据
'''

def dat_to_arrays(dat_path, mdf_name_map=None):
    if mdf_name_map is None:
        mdf_name_map = {}
    mdf = MDF(dat_path)
    # read samples and the timestamps of each variable
    samples_list, timestamps_list, exist_var_name = [], [], []
    print('file name=',dat_path)
    for var_name in config_all.VAR_NAMES:

        # ch_name = mdf_name_map[var_name]
        try:
            signal = mdf.get(var_name)
            samples = signal.samples
            # if len(samples) == 0:
            #     continue
            samples_list.append(samples)
            timestamps_list.append(signal.timestamps)
            exist_var_name.append(var_name)
        except Exception as e:
            print(var_name, e)
            return None

    # incase some signal is zero length
    assert all([len(samples) > 0 for samples in samples_list]), 'some signal is zero length'

    # find the common timestamps
    start, end = max([t[0] for t in timestamps_list]), min([t[-1] for t in timestamps_list])
    time = np.arange(start, end + 1, 1)

    # interpolate the samples
    samples_list = [np.interp(time, timestamps, samples) for timestamps, samples in zip(timestamps_list, samples_list)]

    # so, we have transformed the mdf file into a key2arr dict
    key2arr = dict([*zip(exist_var_name, samples_list)])
    tem_df = pd.DataFrame(key2arr)

    # AGS占空比转化为开度, 0-240线性映射0-90
    tem_df['Ags_Openness'] = (tem_df['sTmSigIn_X_AgsaActPosn'] / 240 * 90).map(int)

    # 计算风阻能耗
    tem_df['C_d'] = (-0.00444 * tem_df['Ags_Openness'] * tem_df['Ags_Openness'] + 0.7467 * tem_df['Ags_Openness'] - 0.1) * 0.001
    tem_df.loc[tem_df['C_d'] < 0, 'C_d'] = 0
    tem_df['V'] = tem_df['COMM_VehicleSpd'] / 3.6
    tem_df['W_Pwr'] = 0.5 * config.W_p * config.W_A * tem_df['C_d'] * tem_df['V'] ** 3

    # 压缩机转速
    tem_df['TmRteOut_CmprSpdReq'] = tem_df['TmRteOut_CmprSpdReq'].map(int)

    # 膨胀阀开度，是否需要分桶？0-100
    tem_df['sTmSigIn_X_CexvActPosn'] = tem_df['sTmSigIn_X_CexvActPosn'].map(int)
    # 鼓风机占空比
    tem_df['Blower_Fr_PWM'] = tem_df['Blower_Fr_PWM'].map(int)
    # 风扇占空比 0-70
    tem_df['CFan_PWM'] = tem_df['CFan_PWM'].map(int)

    # 计算压缩机功率、鼓风机功率、电子风扇功率
    tem_df['Blower_Pwr'] = tem_df['sTmSigIn_I_FrntBlowerLoPwrI'] * tem_df['sTmSigIn_U_FrntBlowerLoPwrU']
    tem_df['Fan_Pwr'] = tem_df['sTmSigIn_I_FanLoPwrI'] * tem_df['sTmSigIn_U_FanLoPwrU']

    # 整体能耗
    tem_df['all_Pwr'] = tem_df['W_Pwr'] + tem_df['TmRteOut_CmprPwrCnsAct'] + tem_df['Blower_Pwr'] + tem_df['Fan_Pwr']
    tem_df['K_all_Pwr'] = tem_df['all_Pwr'].map(int) / 1000

    # 外循环比例
    inlet_pwm = tem_df['Inlet_PWM']
    out_frac = interp_from_table(inlet_pwm, config.inlet_pwm_to_out_frac[:, 0], config.inlet_pwm_to_out_frac[:, 1]) / 100
    tem_df['out_frac'] = out_frac

    # 蒸发器换热前的空气温度 用内外循环加权计算
    temp_cabin = tem_df['TmRteOut_FrntIncarTEstimd']
    temp_amb = tem_df['TmRteOut_HvacAmbTEstimd']
    air_temp_before_heat_exchange = (1 - out_frac) * temp_cabin + out_frac * temp_amb
    tem_df['air_temp_before_heat_exchange'] = air_temp_before_heat_exchange

    # 空调占空比
    hvac_pwm = 100 - tem_df['TmRteOut_FrntBlowerPwmReq']
    tem_df['hvac_pwm'] = hvac_pwm

    '''吹风模式转化为风量'''
    hvac_mode = tem_df['TmRteOut_HvacFirstLeWindModSt']
    all_wind_vol = []
    all_vol_rate = []
    for time_index in range(len(hvac_mode)):
        now_hvac_pwm = hvac_pwm[time_index]
        now_hvac_mode = int(hvac_mode[time_index])

        now_mode_to_vol_rate = config.mode_to_vol_rate[now_hvac_mode]
        now_pwm_to_vol = config.pwm_to_vol[now_hvac_mode]

        k = np.array(list(now_pwm_to_vol.keys()))
        v = np.array(list(now_pwm_to_vol.values()))
        if 15 <= now_hvac_pwm <= 75:
            wind_vol = interp_from_table(now_hvac_pwm, k, v)
        else:
            wind_vol = 0

        k = np.array(list(now_mode_to_vol_rate.keys()))
        v = np.array(list(now_mode_to_vol_rate.values()))[:, config.exist_wind_temp_index]

        if 15 <= now_hvac_pwm <= 75:
            vol_rate = []
            for temp_id in range(len(config.exist_wind_temp_index)):
                # wind_temp_index=config.exist_wind_temp_index[temp_id]
                vol_rate.append(interp_from_table(now_hvac_pwm, k, v[:, temp_id]))
        else:
            vol_rate = np.zeros_like(v[0])

        all_wind_vol.append(wind_vol)
        all_vol_rate.append(vol_rate)

    all_wind_vol = np.array(all_wind_vol)
    all_vol_rate = np.array(all_vol_rate)
    tem_df['wind_vol'] = all_wind_vol

    '''蒸发器换热后的空气温度 用出风口和风量加权计算'''
    # 出风温度
    all_wind_temp = []
    for key_index in range(len(config.all_wind_temp_keys)):
        wind_temp_key = config.all_wind_temp_keys[key_index]
        wind_temp = tem_df[wind_temp_key]
        all_wind_temp.append(wind_temp)

    wind_temp_weight = all_vol_rate / (np.sum(all_vol_rate, axis=-1, keepdims=True) + 1e-10)
    air_temp_after_heat_exchange = np.sum(np.array(all_wind_temp).T * wind_temp_weight, axis=-1)
    tem_df['air_temp_after_heat_exchange'] = air_temp_after_heat_exchange

    return tem_df


if __name__ == '__main__':
    df_list = []
    #raw_data_folder = "./data/energy_cunsum_data"
    raw_data_folder = "data/energy_cunsum_data/bat_file"
    #raw_data_folder = "./raw_data/VP车海南夏季路试/v015"
    out_folder = 'data/energy_cunsum_data/csv_file1'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    all_file_name = os.listdir(raw_data_folder)
    for name_index in tqdm(range(len(all_file_name))):
        file_name = all_file_name[name_index]
        file_path = os.path.join(raw_data_folder, file_name)
        try:
            tem_df = dat_to_arrays(file_path)
            result_path = os.path.join(out_folder, file_name.replace('.dat', '.csv').replace('.mf4', '.csv'))
            tem_df.to_csv(result_path, index=False)

        except Exception as e:
            print(file_path, e)
