import os

import pandas as pd
import numpy as np
import sys

project_folder = os.path.abspath('..')
sys.path.append(os.path.join(project_folder, 'common'))
from common.data_utils_common import interp_from_table, interp_from_table_2d, get_air_density
from common import config_ac


def cal_h(temp_p_h, pressure, states=None):
    log_p = np.log(pressure).reshape(-1, 1)
    log_p_diff = log_p - config_ac.p_h_t_log_p
    log_p_abs_diff = np.abs(log_p_diff)
    min_place = np.argmin(log_p_abs_diff, axis=-1)
    min_place_diff = log_p_diff[np.arange((len(log_p))), min_place]
    min_place_abs_diff = log_p_abs_diff[np.arange((len(log_p))), min_place]

    # 最接近的异号位置
    nearest_next_to_min_place = np.zeros_like(min_place)
    nearest_next_to_min_place[min_place_diff > 0] = min_place[min_place_diff > 0] + 1
    nearest_next_to_min_place[min_place_diff <= 0] = min_place[min_place_diff <= 0] - 1
    nearest_next_to_min_place[nearest_next_to_min_place < 0] = 0
    nearest_next_to_min_place[nearest_next_to_min_place == len(log_p)] = len(log_p) - 1
    nearest_next_to_min_place_abs_diff = log_p_abs_diff[np.arange((len(log_p))), nearest_next_to_min_place]

    # 两次线性插值求解
    min_place_temp_table = config_ac.p_h_t_t[min_place]
    nearest_next_to_min_place_temp_table = config_ac.p_h_t_t[nearest_next_to_min_place]

    # 压强查表权重
    min_place_weight = nearest_next_to_min_place_abs_diff / (min_place_abs_diff + nearest_next_to_min_place_abs_diff)
    min_place_weight = min_place_weight.reshape(-1, 1)
    temp_table = min_place_temp_table * min_place_weight + nearest_next_to_min_place_temp_table * (1 - min_place_weight)

    # 找温差最小的位置对应的焓值
    # 根据物态 筛选范围
    if states == None:
        temp_diff = temp_p_h.reshape(-1, 1) - temp_table
    elif states == 'liquid':
        temp_diff = []
        for data_index in range(len(temp_p_h)):
            now_temp_p_h = temp_p_h[data_index]
            now_temp_table = temp_table[data_index]

            # 找到所有可用的范围
            available_end_index = 0
            for table_data_index in range(len(now_temp_table) - 1):
                now_temp_table_data = now_temp_table[table_data_index]
                next_temp_table_data = now_temp_table[table_data_index + 1]

                if now_temp_table_data == next_temp_table_data:
                    # 多留一个 保证气液混合数据可以用上
                    available_end_index = table_data_index + 1
                    break

            available_table_data = now_temp_table[:available_end_index]
            now_temp_diff = now_temp_p_h - available_table_data

            # pad为一样的长度
            pad_data = np.ones_like(now_temp_table) * np.inf
            pad_data[:len(now_temp_diff)] = now_temp_diff
            temp_diff.append(pad_data)

        temp_diff = np.array(temp_diff)

    elif states == 'gas':
        temp_diff = []
        for data_index in range(len(temp_p_h)):
            now_temp_p_h = temp_p_h[data_index]
            now_temp_table = temp_table[data_index]

            # 找到所有可用的范围
            available_end_index = 0
            for table_data_index in range(len(now_temp_table) - 1):
                now_temp_table_data = now_temp_table[len(now_temp_table) - 1 - table_data_index]
                next_temp_table_data = now_temp_table[len(now_temp_table) - 1 - (table_data_index + 1)]

                if now_temp_table_data == next_temp_table_data:
                    # 多留一个 保证气液混合数据可以用上
                    available_end_index = table_data_index + 1
                    break

            available_table_data = now_temp_table[len(now_temp_table) - available_end_index:]
            now_temp_diff = now_temp_p_h - available_table_data

            # pad为一样的长度
            pad_data = np.ones_like(now_temp_table) * np.inf
            pad_data[len(now_temp_table) - available_end_index:] = now_temp_diff
            temp_diff.append(pad_data)

        temp_diff = np.array(temp_diff)
    # elif states=='gas_liquid':
    #     gas_liquid_temp_table =
    #     temp_diff = temp_p_h.reshape(-1, 1) - gas_liquid_temp_table

    temp_abs_diff = np.abs(temp_diff)
    temp_min_place = np.argmin(temp_abs_diff, axis=-1)
    # temp_min_place_diff = temp_diff[np.arange((len(temp_p_h))), temp_min_place]
    # temp_min_place_abs_diff = temp_abs_diff[np.arange((len(temp_p_h))), temp_min_place]

    '''    
    temp_nearest_next_to_min_place = np.zeros_like(temp_min_place)
    temp_nearest_next_to_min_place[temp_min_place_diff > 0] = temp_min_place[temp_min_place_diff > 0] + 1
    temp_nearest_next_to_min_place[temp_min_place_diff <= 0] = temp_min_place[temp_min_place_diff <= 0] - 1
    temp_nearest_next_to_min_place[temp_nearest_next_to_min_place < 0] = 0
    temp_nearest_next_to_min_place[temp_nearest_next_to_min_place == len(temp_p_h)] = len(temp_p_h) - 1
    temp_nearest_next_to_min_place_abs_diff = temp_abs_diff[np.arange((len(temp_p_h))), nearest_next_to_min_place]

    # 温度查表权重
    min_temp_place_weight = temp_nearest_next_to_min_place_abs_diff / (temp_min_place_abs_diff + temp_nearest_next_to_min_place_abs_diff)

    temp_min_place_h = config_ac.p_h_t_h[temp_min_place]
    temp_nearest_next_to_min_place_h = config_ac.p_h_t_h[temp_nearest_next_to_min_place]
    h_p_h = min_temp_place_weight * temp_min_place_h + (1 - min_temp_place_weight) * temp_nearest_next_to_min_place_h
    '''
    h_p_h = config_ac.p_h_t_h[temp_min_place]

    return h_p_h


def read_csv(data_path, car_type):
    data = pd.read_csv(data_path)

    # 车速 km/h
    car_speed = np.array(data['COMM_VehicleSpd'])

    # 环境温度
    temp_amb = np.array(data['TmRteOut_AmbTEstimd'])


    # 阻力功率 经验估计值
    # resistance_p = np.array(data['W_Pwr'])

    # ags开度 0-90
    # ags_openness = np.array(data['Ags_Openness'])

    # ags硬件开度 0-240
    # ags_act_posn = np.array(data['sTmSigIn_X_AgsaActPosn'])
    # ags硬件开度请求 0-240
    # ags_act_posn_req = np.array(data['AGSPosPeq'])

    # 风扇占空比
    cfan_pwm = np.array(data['CFan_PWM'])
    # 风扇占空比请求
    # cfan_pwm_req = np.array(data['CFan_PWMReq'])

    # 压缩机功率
    compressor_p = np.array(data['TmRteOut_CmprPwrCnsAct'])

    # 加热器功率
    warmer_p = np.array(data['TmRteOut_HvchPwrCns'])

    # 高低压 kpa
    hi_pressure = np.array(data['TmRteOut_HiSideP'])
    lo_pressure = np.array(data['TmRteOut_LoSideP'])

    '''PID相关'''
    # exv_pid_pout = np.array(data['EXV_Oh_PID_Pout'])    # 膨胀阀开度比例输出
    exv_pid_iout = np.array(data['EXV_Oh_PID_Iout'])  # 膨胀阀开度积分输出
    # exv_pid_dout = np.array(data['EXV_Oh_PID_Dout'])    # 膨胀阀开度微分输出
    # exv_pid_out = np.array(data['EXV_Oh_PID_Out'])      # 膨胀阀开度最终PID输出

    # ac_pid_pout_hp = np.array(data['AC_PID_Pout_HP'])   # 压缩机转速比例输出  未找到
    # ac_pid_iout_hp = np.array(data['AC_PID_Iout_HP'])   # 压缩机转速积分输出
    # ac_pid_dout_hp = np.array(data['AC_PID_Dout_HP'])   # 压缩机转速微分输出
    ac_pid_out_hp = np.array(data['AC_PID_out'])  # 压缩机转速最终PID输出

    '''乘员舱请求和设定温度'''
    cab_fl_set_temp = np.array(data['HMI_Temp_FL'])  # 主驾温度设定
    cab_fr_set_temp = np.array(data['HMI_Temp_FR'])  # 副驾温度设定
    cab_rl_set_temp = np.array(data['HMI_Temp_RL'])  # 后排温度设定

    cab_fl_req_temp = np.array(data['DVT_FL'])  # 主驾目标进风温度
    cab_fr_req_temp = np.array(data['DVT_FR'])  # 副驾目标进风温度
    cab_rl_req_temp = np.array(data['DVT_RL'])  # 后排目标进风温度

    '''膨胀阀开度'''
    # 制热条件下乘员舱膨胀阀开度
    cab_heating_status_act_pos = np.array(data['sTmSigIn_X_CexvActPosn'])

    # 制冷条件下乘员舱膨胀阀开度
    cab_cooling_status_act_pos = np.array(data['sTmSigIn_X_EexvActPosn'])

    # 制冷条件下电池膨胀阀开度
    battery_cooling_status_act_pos = np.array(data['sTmSigIn_X_BexvActPosn'])

    # 压缩机转速
    if car_type == 'lemans_low' or car_type == 'lemans_mid':
        if 'TmRteOut_CanCmprSpdReq' in data:
            compressor_speed = np.array(data['TmRteOut_CanCmprSpdReq'])
            if (compressor_speed == compressor_speed[0]).all():
                compressor_speed = np.array(data['TmRteOut_CmprSpdReq'])
        else:
            compressor_speed = np.array(data['TmRteOut_CmprSpdReq'])
    else:
        compressor_speed = np.array(data['TmRteOut_CmprSpdReq'])

    # 热泵模式
    hp_mode = np.array(data['HP_Mode_Valve'])

    # 空调模式
    hp_mode_ac = np.array(data['HP_Mode_AC'])

    # 空调乘员舱吹风模式
    hvac_mode_first = np.array(data['hvac_mode_first'])
    hvac_mode_second = np.array(data['hvac_mode_second'])
    if 'hvac_mode' in data:
        hvac_mode = np.array(data['hvac_mode'])
    else:
        hvac_mode = np.array([hvac_mode_first, hvac_mode_second]).T

    # 换热前后空调风温度
    # air_temp_before_heat_exchange = np.array(data['air_temp_before_heat_exchange'])
    # air_temp_after_heat_exchange = np.array(data['air_temp_after_heat_exchange'])

    # 风量
    wind_vol = np.array(data['wind_vol'])
    # 鼓风机pwm
    hvac_pwm = np.array(data['hvac_pwm'])

    '''压缩机制冷剂在压焓图上的相关温度'''
    temp_p_h_2 = np.array(data['TmRteOut_CmprDchaT'])
    temp_p_h_5 = np.array(data['TmRteOut_InrCondOutlT'])
    # 制冷剂低压温度 6-7
    lo_pressure_temp = interp_from_table(lo_pressure, config_ac.r134_p_t[:, 1], config_ac.r134_p_t[:, 0])

    # 制冷剂高压温度 3-4
    hi_pressure_temp = interp_from_table(hi_pressure, config_ac.r134_p_t[:, 1], config_ac.r134_p_t[:, 0])

    # # 制冷剂乘员舱端 制冷 压焓图1处温度
    # temp_p_h_1_cab_cooling = np.array(data['TmRteOut_FrntEvaprOutlT'])
    #
    # # 制冷剂电池端 制冷 压焓图1处温度
    # temp_p_h_1_battery_cooling = np.array(data['TmRteOut_ChllrOutlT'])

    # 制冷剂乘员舱端 制热 压焓图1处温度
    temp_p_h_1_cab_heating = np.array(data['TmRteOut_OutrCondOutlT'])

    # 压焓图5的焓值
    h_p_h_5 = cal_h(temp_p_h_5, hi_pressure, states='liquid')

    # 压焓图1的焓值
    # h_p_h_1_cab_cooling = cal_h(temp_p_h_1_cab_cooling, lo_pressure, states='gas')
    # 制热模式压焓图1的焓值
    h_p_h_1_cab_heat = cal_h(temp_p_h_1_cab_heating, lo_pressure, states='gas')

    # 压缩机排量
    if car_type == 'modena':
        data['cmpr_capacity'] = config_ac.compressor_displacement_modena
    elif car_type == 'lemans_adv':
        data['cmpr_capacity'] = config_ac.compressor_displacement_lemans_adv
    elif car_type == 'lemans_low':
        data['cmpr_capacity'] = config_ac.compressor_displacement_lemans_low
    elif car_type == 'lemans_mid':
        data['cmpr_capacity'] = config_ac.compressor_displacement_lemans_mid
    cmpr_capacity = np.array(data['cmpr_capacity'])

    # 压焓图6的气液比例
    key_1 = np.log(lo_pressure)
    key_2 = h_p_h_5
    table_key_1 = config_ac.p_h_x_log_p
    table_key_2 = config_ac.p_h_x_h
    table_value = config_ac.p_h_x_x
    p_h_6_g_l_rate = interp_from_table_2d(key_1, key_2, table_key_1, table_key_2, table_value)

    '''冷媒相关'''
    if car_type == 'lemans_low' or car_type == 'lemans_mid':
        if 'sTmSigIn_Te_CanHvchOutCooltT' in data:
            hvch_cool_medium_temp_out = np.array(data['sTmSigIn_Te_CanHvchOutCooltT'])
            if (hvch_cool_medium_temp_out == hvch_cool_medium_temp_out[0]).all():
                hvch_cool_medium_temp_out = np.array(data['sTmSigIn_Te_CanHvchOutCooltT'])
        else:
            hvch_cool_medium_temp_out = np.array(data['sTmSigIn_Te_HvchOutCooltT'])

        if 'sTmSigIn_Te_CanHvchInCooltT' in data:
            hvch_cool_medium_temp_in = np.array(data['sTmSigIn_Te_CanHvchInCooltT'])
            if (hvch_cool_medium_temp_in == hvch_cool_medium_temp_in[0]).all():
                hvch_cool_medium_temp_in = np.array(data['sTmSigIn_Te_CanHvchInCooltT'])
        else:
            hvch_cool_medium_temp_in = np.array(data['sTmSigIn_Te_HvchInCooltT'])

        # hvch_cool_medium_temp_out = np.array(data['sTmSigIn_Te_CanHvchOutCooltT'])
        # hvch_cool_medium_temp_in = np.array(data['sTmSigIn_Te_CanHvchInCooltT'])
    else:
        hvch_cool_medium_temp_out = np.array(data['sTmSigIn_Te_HvchOutCooltT'])
        hvch_cool_medium_temp_in = np.array(data['sTmSigIn_Te_HvchInCooltT'])

    # 过冷度 过热度
    sc_act_mode_10 = np.array(data['EXV_Oh_SC_Act'])  # 模式为10实际过冷度
    sh_act_mode_10 = np.array(data['EXV_Oh_SH_Act'])  # 模式为10实际过热度
    sc_tar_mode_10 = np.array(data['EXV_Oh_SC_Tar'])  # 模式为10目标过冷度
    sh_tar_mode_10 = np.array(data['EXV_Oh_SH_Tar'])  # 模式为10目标过热度

    # 气体换热量
    # 乘员舱气体换热量单位j 乘以1e-3保证单位一至
    # wind_heat_exchange = (air_temp_after_heat_exchange - air_temp_before_heat_exchange) * wind_vol * get_air_density(
    #     air_temp_after_heat_exchange) * config_ac.c_air * 1e-3
    # wind_heat_exchange = np.abs(wind_heat_exchange)
    # 乘员舱气体换热表征制冷剂物态换热 可以求解制冷剂流量 kg/s
    # cab_refrigerant_vol = np.abs(wind_heat_exchange / (h_p_h_1_cab_cooling - h_p_h_5))

    # 查表计算流量
    # volumetric_efficiency = config_ac.volumetric_efficiency_dict
    table_speed = np.array(list(config_ac.volumetric_efficiency_dict.keys()))
    speed_diff = compressor_speed.reshape(-1, 1) - table_speed
    speed_diff_abs = np.abs(speed_diff)
    min_place_speed_diff = np.argmin(speed_diff_abs, axis=-1)
    use_speed = (np.repeat(table_speed.reshape(1, -1), len(speed_diff_abs), axis=0))[
        np.arange(len(min_place_speed_diff)), min_place_speed_diff]
    pressure_rate = hi_pressure / lo_pressure
    all_volumetric_efficiency = []
    for data_index in range(len(use_speed)):
        now_use_speed = use_speed[data_index]
        now_pressure_rate = pressure_rate[data_index]
        now_pressure_rate_dict = config_ac.volumetric_efficiency_dict[now_use_speed]
        table_keys = np.array(list(now_pressure_rate_dict.keys()))
        table_values = np.array(list(now_pressure_rate_dict.values())) / 100

        if now_pressure_rate < np.min(table_keys):
            volumetric_efficiency = table_values[np.argmin(table_keys)]
        elif now_pressure_rate > np.max(table_keys):
            volumetric_efficiency = table_values[np.argmax(table_keys)]
        else:
            volumetric_efficiency = interp_from_table(now_pressure_rate, table_keys, table_values)

        all_volumetric_efficiency.append(volumetric_efficiency)

    all_volumetric_efficiency = np.array(all_volumetric_efficiency)
    total_refrigerant_vol = cmpr_capacity * compressor_speed * 60 / 1000000 * all_volumetric_efficiency

    '''控制目标相关'''
    # Hvch出口目标水温
    dwt = np.array(data['DWT_DWT'])
    # dht
    dht = np.array(data['DHT_DHT'])
    # 目标饱和高压
    aim_hi_pressure = np.array(data['AC_HP_Tar'])
    # 目标饱和高压
    aim_lo_pressure = np.array(data['AC_LP_Target'])

    '''水泵相关'''
    # 制热水泵pwm
    heat_pump_pwm = np.array(data['sTmSigIn_X_HcwpActSpd'])
    # 制热端水泵流量
    # heat_coolant_vol = np.array(data['Pump_Heat_ActFlow'])
    # 电池端水泵流量
    # battery_coolant_vol = np.array(data['Pump_Batt_ActFlow'])
    # 电驱水泵流量
    # motor_coolant_vol = np.array(data['Pump_Motor_ActFlow'])

    # 进电池前，和空调换热后电池冷却液温度
    temp_coolant_battery_in = np.array(data['TmRteOut_HvBattInletCooltT'])
    # 进电池后，和空调换热前电池冷却液温度
    temp_coolant_battery_out = np.array(data['TmRteOut_HvBattOutletCooltT'])
    # 电驱回路出口冷却液温度
    # temp_coolant_motor_out = np.array(data['IPU_CoolMaxT'])

    bctv_act_posn = np.array(data['sTmSigIn_X_BctvActPosn'])
    wt_mode_9way = np.array(data['WT_Mode_9Way'])

    # AC_PID_相关
    # AC_KpRate
    ac_kp_rate = np.array(data['AC_KpRate'])
    # 电池请求冷却的温度
    temp_battery_req = np.array(data['HP_Mode_BattWTReqOpt'])
    # 乘员舱温度 原始
    # temp_in_car = np.array(data['TmRteOut_FrntIncarTRaw'])
    # 当前乘务舱温度 滤波后的值
    temp_in_car = np.array(data['TmRteOut_FrntIncarTEstimd'])
    # 蒸发器温度
    temp_evap = np.array(data['SEN_Evap_Fr'])

    '''低压能耗'''
    # 鼓风机功率
    # blower_p = np.array(data['Blower_Pwr'])

    # 电子风扇功率
    # fan_p = np.array(data['Fan_Pwr'])

    blower_p = np.array(data['POW_LV_Blower'])  # 鼓风机功率
    pump_hvh_p = np.array(data['POW_LV_Pump_HVH'])  # 暖芯水泵功耗
    # pump_motor_p = np.array(data['POW_LV_Pump_Motor'])  # 电驱水泵功耗
    # pump_batt_p = np.array(data['POW_LV_Pump_Batt'])  # 电池水泵功耗
    fan_p = np.array(data['POW_LV_Cfan'])  # 冷却风扇功耗

    out_dict = {
        'car_speed': car_speed,
        'temp_amb': temp_amb,

        'compressor_p': compressor_p,

        'blower_p': blower_p,
        'fan_p': fan_p,
        'pump_hvh_p': pump_hvh_p,
        # 'pump_motor_p': pump_motor_p,
        # 'pump_batt_p': pump_batt_p,

        # 'ags_openness': ags_openness,
        # 'ags_act_posn': ags_act_posn,
        # 'ags_act_posn_req': ags_act_posn_req,
        'cfan_pwm': cfan_pwm,
        # 'cfan_pwm_req': cfan_pwm_req,
        # 'all_p': all_p,
        'warmer_p': warmer_p,
        'hi_pressure': hi_pressure,
        'lo_pressure': lo_pressure,
        'cab_heating_status_act_pos': cab_heating_status_act_pos,
        'cab_cooling_status_act_pos': cab_cooling_status_act_pos,
        'battery_cooling_status_act_pos': battery_cooling_status_act_pos,
        'compressor_speed': compressor_speed,
        'hp_mode': hp_mode,
        'hp_mode_ac': hp_mode_ac,
        'hvac_mode_first': hvac_mode_first,
        'hvac_mode_second': hvac_mode_second,
        'hvac_mode': hvac_mode,
        'lo_pressure_temp': lo_pressure_temp,
        'hi_pressure_temp': hi_pressure_temp,
        # 'air_temp_before_heat_exchange': air_temp_before_heat_exchange,
        # 'air_temp_after_heat_exchange': air_temp_after_heat_exchange,
        'wind_vol': wind_vol,
        'hvac_pwm': hvac_pwm,
        'temp_p_h_2': temp_p_h_2,
        'temp_p_h_5': temp_p_h_5,
        # 'temp_p_h_1_cab_cooling': temp_p_h_1_cab_cooling,
        # 'temp_p_h_1_battery_cooling': temp_p_h_1_battery_cooling,
        'temp_p_h_1_cab_heating': temp_p_h_1_cab_heating,

        'hvch_cool_medium_temp_out': hvch_cool_medium_temp_out,
        'hvch_cool_medium_temp_in': hvch_cool_medium_temp_in,

        'sc_act_mode_10': sc_act_mode_10,
        'sh_act_mode_10': sh_act_mode_10,
        'sc_tar_mode_10': sc_tar_mode_10,
        'sh_tar_mode_10': sh_tar_mode_10,

        'h_p_h_5': h_p_h_5,
        # 'h_p_h_1_cab_cooling': h_p_h_1_cab_cooling,
        'p_h_6_g_l_rate': p_h_6_g_l_rate,

        # 'cab_refrigerant_vol': cab_refrigerant_vol,
        # 'wind_heat_exchange': wind_heat_exchange,
        'total_refrigerant_vol': total_refrigerant_vol,

        'dwt': dwt,
        'dht': dht,
        'aim_hi_pressure': aim_hi_pressure,
        'aim_lo_pressure': aim_lo_pressure,

        'heat_pump_pwm': heat_pump_pwm,
        # 'heat_coolant_vol': heat_coolant_vol,
        # 'battery_coolant_vol': battery_coolant_vol,
        # 'motor_coolant_vol': motor_coolant_vol,

        'temp_coolant_battery_in': temp_coolant_battery_in,
        'temp_coolant_battery_out': temp_coolant_battery_out,
        # 'temp_coolant_motor_out': temp_coolant_motor_out,
        'cmpr_capacity': cmpr_capacity,

        # 'exv_pid_pout' : exv_pid_pout,
        'exv_pid_iout': exv_pid_iout,
        # 'exv_pid_dout': exv_pid_dout,
        # 'exv_pid_out': exv_pid_out,

        # 'ac_pid_pout_hp': ac_pid_pout_hp,  #未找到
        # 'ac_pid_iout_hp': ac_pid_iout_hp,
        # 'ac_pid_dout_hp': ac_pid_dout_hp,
        'ac_pid_out_hp': ac_pid_out_hp,

        'cab_fl_set_temp': cab_fl_set_temp,
        'cab_fr_set_temp': cab_fr_set_temp,
        'cab_rl_set_temp': cab_rl_set_temp,

        'cab_fl_req_temp': cab_fl_req_temp,
        'cab_fr_req_temp': cab_fr_req_temp,
        'cab_rl_req_temp': cab_rl_req_temp,
        'h_p_h_1_cab_heat': h_p_h_1_cab_heat,
        # 压缩机相关
        'temp_in_car': temp_in_car,
        'temp_evap': temp_evap,
        'temp_battery_req': temp_battery_req,
        'ac_kp_rate': ac_kp_rate,

        'bctv_act_posn': bctv_act_posn,
        'wt_mode_9way': wt_mode_9way,
    }

    '''风门相关'''
    mix_door_pwm_fl = np.array(data['MixDor_L_PWM'])  # 主驾驶风门pwm
    mix_door_pwm_fr = np.array(data['MixDor_R_PWM'])  # 副驾驶风门pwm
    mix_door_pwm_dict = {
        'mix_door_pwm_fl': mix_door_pwm_fl,
        'mix_door_pwm_fr': mix_door_pwm_fr,
    }
    out_dict.update(mix_door_pwm_dict)
    try:
        mix_door_pwm_rl = np.array(data['MixDor_RL_PWM'])  # 后排风门pwm
        mix_door_pwm_dict['mix_door_pwm_rl'] = {
            'mix_door_pwm_rl': mix_door_pwm_rl,
        }
        out_dict.update(mix_door_pwm_dict)
    except Exception as e:
        print(e)

    return out_dict
