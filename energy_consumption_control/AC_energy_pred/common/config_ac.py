import os
import pickle

from common import config_common

import numpy as np

'''变量名字'''
base_VAR_NAMES = config_common.common_VAR_NAMES + [
    # 'COMM_VehicleSpd',  # 车速
    'sTmSigIn_X_AgsaActPosn',  # AGS开度
    'AGSPosPeq',  # 默认策略的ags请求0-240
    # 'SubLINR_AgsaActPosn',
    # 'TmRteOut_FanPwmReq',  # 前端风扇的占空比
    # 'TmRteOut_AmbTEstimd',  # 环境温度
    # 'TmRteOut_FrntIncarTEstimd',  # 车内温度
    # 'Inlet_PWM',  # 内外循环pwm
    # 'Outlet_PWM_Defrost',
    # 'Outlet_PWM_Rr',
    'TmRteOut_CmprSpdReq',  # 压缩机转速
    'TmRteOut_CanCmprSpdReq',  # 压缩机转速 lemans中低配可能是这个
    'TmRteOut_CmprPwrCnsAct',  # 压缩机功率
    'TmRteOut_HiSideP',  # 压缩机高压压强
    'TmRteOut_LoSideP',  # 压缩机低压压强

    'sTmSigIn_X_CexvActPosn',  # 乘员舱 膨胀阀开度 制热
    'sTmSigIn_X_EexvActPosn',  # 乘员舱 膨胀阀开度 制冷
    'sTmSigIn_X_BexvActPosn',  # 电池 膨胀阀开度 制冷
    # 'Blower_Fr_PWM',
    'CFan_PWM',  # ags冷却风扇转速
    'CFan_PWMReq',  # ags冷却风扇转速请求
    # 'sTmSigIn_I_FrntBlowerLoPwrI',  # 鼓风机电流
    # 'sTmSigIn_U_FrntBlowerLoPwrU',  # 鼓风机电压
    # 'sTmSigIn_I_FanLoPwrI',  # ags冷却风扇电流
    # 'sTmSigIn_U_FanLoPwrU',  # ags冷却风扇电压
    'sTmSigIn_X_BctvActPosn',  # BCTV模式

    'TmRteOut_FrntEvaprOutlT',  # 压焓图-1 乘员舱 制冷
    'TmRteOut_OutrCondOutlT',  # 压焓图-1 乘员舱 制热
    'TmRteOut_ChllrOutlT',  # 压焓图-1 电池端 制冷
    'TmRteOut_CmprDchaT',  # 压焓图-2
    'TmRteOut_InrCondOutlT',  # 压焓图-5

    # 'HP_Mode_Valve',  # 热泵模式 HP_Mode_AC的子模式
    'HP_Mode_AC',  # 空调模式

    # 'HP_Deice_ACTSHHi',  # 除冰相关

    'POW_LV_Blower',  # 鼓风机功率
    'POW_LV_Pump_HVH',  # 暖芯水泵功耗
    'POW_LV_Pump_Motor',  # 电驱水泵功耗
    'POW_LV_Pump_Batt',  # 电池水泵功耗
    'POW_LV_Cfan',  # 冷却风扇功耗

    # 'TmRteOut_HvacFirstLeWindModSt',  # 乘员舱吹风模式

    # 'TmRteOut_FrntBlowerPwmReq',  # 100-空调占空比

    'TmRteOut_HvBattInletCooltT',  # 进电池前 电池冷却液温度
    'TmRteOut_HvBattOutletCooltT',  # 进电池后 电池冷却液温度

    'IPU_CoolMaxT',  # 电驱回路出口冷却液温度

    'Pump_Bat_PWM_Req',  # 电池水泵占空比
    'WT_Mode_9Way',  # 电池水泵模式

    'TmRteOut_HvchPwrCns',  # 加热器功率,PTC功率
    'sTmSigIn_Te_HvchOutCooltT',  # 制热时冷却液PTC换热后温度
    'sTmSigIn_Te_HvchInCooltT',  # 制热时冷却液lcc换热后 PTC换热前
    'sTmSigIn_Te_CanHvchOutCooltT',  # lemans-1/-2制热时冷却液PTC换热后温度
    'sTmSigIn_Te_CanHvchInCooltT',  # lemans-1/-2制热时冷却液PTC换热前温度
    # 'DWT_DWT',  # Hvch出口目标水温
    # 'DHT_DHT',  # DHT,控制量
    'AC_HP_Tar',  # 目标饱和高压
    'AC_LP_Target',  # 目标饱和低压

    'sTmSigIn_X_HcwpActSpd',  # 暖芯水磅转速pwm

    'Pump_Heat_ActFlow',  # 制热端水泵流量
    'Pump_Batt_ActFlow',  # 电池端水泵流量
    'Pump_Motor_ActFlow',  # 电驱水泵流量

    # 'EXV_Oh_SC_Err',  # 模式为10的过冷度偏差 （实际控制压缩机转速的量）
    'EXV_Oh_SC_Act',  # 模式为10实际过冷度
    'EXV_Oh_SH_Act',  # 模式为10实际过热度
    'EXV_Oh_SC_Tar',  # 模式为10目标过冷度
    'EXV_Oh_SH_Tar',  # 模式为10目标过热度

    # 'EXV_Oh_PID_Pout',  # 膨胀阀开度比例输出
    'EXV_Oh_PID_Iout',  # 膨胀阀开度积分输出
    # 'EXV_Oh_PID_Dout',  # 膨胀阀开度微分输出
    # 'EXV_Oh_PID_Out',  # 膨胀阀开度最终PID输出

    # 高压部分
    # 'AC_PID_Pout_HP',  # 压缩机转速比例输出
    # 'AC_PID_Iout_HP',  # 压缩机转速积分输出
    # 'AC_PID_Dout_HP',  # 压缩机转速微分输出
    'AC_PID_out',  # 压缩机转速最终PID输出

    # 'HMI_Temp_FL',  # 主驾温度设定
    # 'HMI_Temp_FR',  # 主驾温度设定
    # 'HMI_Temp_RL',  # 后排温度设定
    #
    # 'DVT_FL',  # 主驾目标进风温度
    # 'DVT_FR',  # 副驾目标进风温度
    # 'DVT_RL',  # 后排目标进风温度

    'HP_Mode_BattWTReqOpt',  # 电池请求冷却的温度
    'AC_KpRate',  # AC_PID_out PID系数

] + config_common.all_wind_channel_temp_keys

VAR_NAMES_modena = base_VAR_NAMES + config_common.mix_door_pwm_list_modena
VAR_NAMES_lemans = base_VAR_NAMES + config_common.mix_door_pwm_list_lemans + config_common.lemans_other

# '''阻力相关'''
# W_A_modena = 2.43  # 迎风面积
# W_A_lemans = 2.47  # 迎风面积
# W_p = 1.2  # 空气密度
# zero_temp_bias = -273.15  # 绝对0度
# zero_air_density = 1.293  # 标准气压的0度的空气密度
c_air = 1005  # 空气比热容
# c_r134a = 3300  # r134a比热容

cool_medium_temp_in_padding = -1000.0  # 进水口水温为多少是padding

'''训练相关'''
lr = 1e-3
train_batch_size = 1024
train_epoch = 50
eval_batch_size = 1
gard_scale = 1
min_split_len = 100
base_dataset_folder = './data/datset_pkl/'

'''转化所需的表'''

# 制冷剂温度vs饱和压力
r134_p_t = np.array([[-62, 13.9], [-60, 15.9], [-58, 18.1], [-56, 20.5], [-54, 23.2],
                     [-52, 26.2], [-50, 29.5], [-48, 33.1], [-46, 37.0], [-44, 41.3],
                     [-42, 46.1], [-40, 51.2], [-38, 56.8], [-36, 62.9], [-34, 69.5],
                     [-32, 76.7], [-30, 84.4], [-28, 92.7], [-26, 101.7], [-24, 111.3],
                     [-22, 121.6], [-20, 132.7], [-18, 144.6], [-16, 157.3], [-14, 170.8],
                     [-12, 185.2], [-10, 200.6], [-8, 216.9], [-6, 234.3], [-4, 252.7],
                     [-2, 272.2], [0, 292.8], [2, 314.6], [4, 337.7], [6, 362.0],
                     [8, 387.6], [10, 414.6], [12, 443.0], [14, 472.9], [16, 504.3],
                     [18, 537.2], [20, 571.7], [22, 607.9], [24, 645.8], [26, 685.4],
                     [28, 726.9], [30, 770.2], [32, 815.4], [34, 862.6], [36, 911.8],
                     [38, 963.2], [40, 1016.6], [42, 1072.2], [44, 1130.1], [46, 1190.3],
                     [48, 1252.9], [50, 1317.9], [52, 1385.4], [54, 1455.5], [56, 1528.2],
                     [58, 1603.6], [60, 1681.8], [62, 1762.8], [64, 1846.7], [66, 1933.7],
                     [68, 2023.7], [70, 2116.8], [72, 2213.2], [74, 2313.0], [76, 2416.1],
                     [78, 2522.8], [80, 2633.2], [82, 2747.3], [84, 2865.3], [86, 2987.4],
                     [88, 3113.6], [90, 3244.2], [92, 3379.3], [94, 3519.3], [96, 3664.5]])

# 气液混合比例相关
now_path = os.path.dirname(__file__)
gas_liquid_mixing_ratio_table_path = os.path.join(now_path, '../auto_get_data_from_pic/data/pkl/p_h_x_table_dict.pkl')
with open(gas_liquid_mixing_ratio_table_path, 'rb') as f:
    gas_liquid_table_dict = pickle.load(f)

p_h_x_log_p = gas_liquid_table_dict['log_p']
p_h_x_h = gas_liquid_table_dict['h']
p_h_x_x = gas_liquid_table_dict['x']

# 热熵温度表
p_h_t_table_path = os.path.join(now_path, '../auto_get_data_from_pic/data/pkl/p_h_t_table_dict.pkl')
with open(p_h_t_table_path, 'rb') as f:
    p_h_t_table_dict = pickle.load(f)

p_h_t_log_p = p_h_t_table_dict['log_p']
p_h_t_h = p_h_t_table_dict['h']
p_h_t_t = p_h_t_table_dict['t']

'''流量相关查表'''
# 压缩机排量 单位cc ml^3
compressor_displacement_modena = 33
compressor_displacement_lemans_low = 33
compressor_displacement_lemans_mid = 33
compressor_displacement_lemans_adv = 45

# 容积效率表 rpm 高低压比 容积效率
volumetric_efficiency_dict = {
    800: {
        2.5: 85.5,
        3: 88.1,
        4: 85.8,
    },
    1000: {
        3: 90.6,
        3.428571429: 89.4,
        4: 87.9,
    },
    1500: {
        3.428571429: 90.9,
    },
    2000: {
        3.428571429: 91.7,
        4: 92,
        5: 89.5,
    },
    3000: {
        4: 92.5,
        5: 91.4,
        6: 89.7,
    },
    4000: {
        5: 92.9,
        6: 91.6,
        7: 89.5,
    },
    6000: {
        5: 94.9,
        6: 93.9,
        7: 93.3,
    },
    8000: {
        5: 96.6,
        6: 94.9,
        7: 94
    },
    12000: {
        6: 91.7,
        7: 91.4,
        8: 90.9,
    },
}
