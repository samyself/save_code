import pickle

import numpy as np

'''变量名字'''
all_wind_temp_keys = [
    'TmRteOut_HvacFirstLeVentnAirT',  # 0
    'TmRteOut_HvacFirstRiVentnAirT',  # 1
    'TmRteOut_HvacFirstLeFlrAirT',  # 2
    'TmRteOut_HvacFirstRiFlrAirT',  # 3
]

VAR_NAMES = [
                'COMM_VehicleSpd',  # 车速
                'sTmSigIn_X_AgsaActPosn',  # AGS开度
                # 'AGSPosPeq',  # 默认策略的ags请求0-240
                'SubLINR_AgsaActPosn',
                'TmRteOut_FanPwmReq',  # 前端风扇的占空比
                'TmRteOut_AmbTEstimd',  # 环境温度
                'TmRteOut_HvacAmbTEstimd',
                'TmRteOut_FrntIncarTEstimd',  # 车内温度
                'Inlet_PWM',  # 内外循环pwm
                'Outlet_PWM_Defrost',
                'Outlet_PWM_Rr',
                'TmRteOut_CmprSpdReq',  # 压缩机转速
                'TmRteOut_CmprPwrCnsAct',  # 压缩机功率
                'TmRteOut_HiSideP',  # 压缩机高压压强
                'TmRteOut_LoSideP',  # 压缩机低压压强

                'sTmSigIn_X_CexvActPosn',  # 乘员舱 膨胀阀开度 制热
                'sTmSigIn_X_EexvActPosn',  # 乘员舱 膨胀阀开度 制冷
                'sTmSigIn_X_BexvActPosn',  # 电池 膨胀阀开度 制冷
                'Blower_Fr_PWM',
                'CFan_PWM',  # ags冷却风扇转速
                # 'CFan_PWMReq',  # ags冷却风扇转速请求
                'sTmSigIn_I_FrntBlowerLoPwrI',  # 鼓风机电流
                'sTmSigIn_U_FrntBlowerLoPwrU',  # 鼓风机电压
                'sTmSigIn_I_FanLoPwrI',  # ags冷却风扇电流
                'sTmSigIn_U_FanLoPwrU',  # ags冷却风扇电压

                # 电阻转温度
                # 'Subsen_ChllrOutlTSnsrU', #低压温度（Chiller）压焓图-1 电池端
                # 'Subsen_FrntEvaprOutlTSnsrU', #低压温度（蒸发器）压焓图-1乘员舱
                # 'TmRteOut_InrCondOutlTSnsrU', #高压温度  压焓图-5

                'TmRteOut_FrntEvaprOutlT',  # 压焓图-1 乘员舱 制冷
                'TmRteOut_OutrCondOutlT',  # 压焓图-1 乘员舱 制热
                'TmRteOut_ChllrOutlT',  # 压焓图-1 电池端 制冷
                'TmRteOut_CmprDchaT',  # 压焓图-2
                'TmRteOut_InrCondOutlT',  # 压焓图-5

                'HP_Mode_Valve',  # 热泵模式 HP_Mode_AC的子模式
                'HP_Mode_AC',  # 空调模式

                # 'HP_Deice_ACTSHHi',  # 除冰相关

                # 蒸发器换热前的空气温度 实车没有数据 暂时不用 用内外循环计算
                # 'Tvent_Evp_in1',
                # 'Tvent_Evp_in2',
                # 'Tvent_Evp_in3',
                # 'Tvent_Evp_in4',

                'TmRteOut_HvacFirstLeWindModSt',  # 乘员舱吹风模式
                'TmRteOut_FrntBlowerPwmReq',  # 100-空调占空比

                'TmRteOut_HvBattInletCooltT',  # 进电池前 电池冷却液温度
                'TmRteOut_HvBattOutletCooltT',  # 进电池后 电池冷却液温度

                'IPU_CoolMaxT',  # 电驱回路出口冷却液温度

                'Pump_Bat_PWM_Req',  # 电池水泵占空比
                'WT_Mode_9Way',  # 电池水泵模式

                'TmRteOut_HvchPwrCns',  # 加热器功率,PTC功率
                'sTmSigIn_Te_HvchOutCooltT',  # 制热时冷却液PTC换热后温度
                'sTmSigIn_Te_HvchInCooltT',  # 制热时冷却液lcc换热后 PTC换热前
                'DWT_DWT',  # Hvch出口目标水温
                'DHT_DHT',  # DHT,控制量
                'AC_HP_Tar',  # 目标饱和高压
                'AC_LP_Target',  # 目标饱和低压

                'sTmSigIn_X_HcwpActSpd',  # 暖芯水磅转速
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

                # 'DVT_FL',  # 主驾目标进风温度
                # 'DVT_FR',  # 副驾目标进风温度
                # 'DVT_RL',  # 后排目标进风温度

                'HP_Mode_BattWTReqOpt',  # 电池请求冷却的温度'

                'AC_KpRate'

            ] + all_wind_temp_keys

'''阻力相关'''
W_A = 2.43  # 迎风面积
W_p = 1.2  # 空气密度
zero_temp_bias = -273.15  # 绝对0度
zero_air_density = 1.293  # 标准气压的0度的空气密度
c_air = 1005  # 空气比热容
c_r134a = 3300  # r134a比热容

'''加热制冷模式相关参数'''
# 模式名字
cooling_mode_index = 2
heat_mode_index = 10

# 风道参数 换算为dm 风量为L/s
channel_s = 0.35  # dm^2
channel_cooling_place = 5  # dm
channel_heating_place = 8  # dm

'''训练相关'''
lr = 1e-2
train_batch_size = 1024
train_epoch = 20
eval_batch_size = 1
gard_scale = 1
min_split_len = 100

'''转化所需的表'''
# 空调乘员舱 占空比转化为总风量
pwm_to_vol = {
    0: {
        15: 0,
        75: 0,
    },
    1: {
        15: 73.22,
        20: 114.02,
        25: 157.19,
        30: 200.65,
        35: 243.76,
        40: 288.73,
        45: 329.38,
        50: 371.59,
        55: 405.01,
        60: 434.41,
        65: 465.59,
        70: 494.78,
        75: 509.78,
    },  # Face
    2: {
        15: 56.17,
        20: 80.16,
        25: 109.4,
        30: 138.2,
        35: 170.5,
        40: 201.49,
        45: 234.53,
        50: 267.26,
        55: 309,
        60: 336.46,
        65: 371.8,
        70: 406.62,
        75: 441.6
    },  # Foot
    3: {
        15: 53.41,
        20: 79.34,
        25: 108.82,
        30: 138.44,
        35: 172.79,
        40: 208.48,
        45: 244.7,
        50: 280.57,
        55: 318.08,
        60: 356.92,
        65: 393.36,
        70: 419.48,
        75: 449.73,
    },  # Faceandfoot
    4: {
        15: 54.86,
        20: 78.9,
        25: 106.58,
        30: 131.96,
        35: 162.54,
        40: 192.72,
        45: 224.01,
        50: 253.8,
        55: 285.43,
        60: 319.46,
        65: 352.27,
        70: 384.57,
        75: 408.99
    },  # Defrost
    5: {
        15: 0,
        20: 0,
        25: 0,
        30: 0,
        35: 0,
        40: 0,
        45: 0,
        50: 0,
        55: 0,
        60: 0,
        65: 0,
        70: 0,
        75: 0
    },  # Faceanddefrost
    6: {
        15: 50.24,
        20: 74.45,
        25: 100.81,
        30: 127.31,
        35: 157.17,
        40: 186.3,
        45: 217.44,
        50: 247.39,
        55: 279.76,
        60: 314.35,
        65: 348.05,
        70: 379.24,
        75: 405.75
    },  # Footanddefrost
    7: {
        15: 59.04,
        20: 86.88,
        25: 114.78,
        30: 148.63,
        35: 185.89,
        40: 222.5,
        45: 260.28,
        50: 301.08,
        55: 341.39,
        60: 384.05,
        65: 418.49,
        70: 447,
        75: 476.5,
    },  # Faceandfootanddefrost
}

# 实车有的风道温度
exist_wind_temp_index = np.array([0, 2, 4, 5])

# 风量比例
mode_to_vol_rate = {
    0: {
        0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        100: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    },
    1: {
        15: np.array([0.171, 0.207, 0, 0.202, 0.183, 0, 0.237, 0, 0, 0, 0, 0]),
        30: np.array([0.175, 0.203, 0, 0.201, 0.178, 0, 0.243, 0, 0, 0, 0, 0]),
        45: np.array([0.177, 0.206, 0, 0.201, 0.179, 0, 0.237, 0, 0, 0, 0, 0]),
        60: np.array([0.179, 0.203, 0, 0.206, 0.178, 0, 0.234, 0, 0, 0, 0, 0]),
        75: np.array([0.179, 0.209, 0, 0.206, 0.179, 0, 0.227, 0, 0, 0, 0, 0]),
    },
    2: {
        15: np.array([0.072, 0, 0.146, 0, 0.090, 0.159, 0, 0.148, 0.173, 0.148, 0.024, 0.040]),
        30: np.array([0.056, 0, 0.163, 0, 0.063, 0.163, 0, 0.169, 0.179, 0.152, 0.024, 0.029]),
        45: np.array([0.059, 0, 0.173, 0, 0.059, 0.163, 0, 0.173, 0.176, 0.149, 0.023, 0.026]),
        60: np.array([0.059, 0, 0.178, 0, 0.059, 0.169, 0, 0.177, 0.176, 0.131, 0.024, 0.027]),
        75: np.array([0.057, 0, 0.178, 0, 0.055, 0.187, 0, 0.174, 0.171, 0.126, 0.025, 0.027]),
    },
    3: {
        0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        100: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    },
    4: {
        15: np.array([0.096, 0, 0, 0, 0.011, 0, 0, 0, 0, 0.651, 0.072, 0.072]),
        30: np.array([0.066, 0, 0, 0, 0.066, 0, 0, 0, 0, 0.716, 0.076, 0.076]),
        45: np.array([0.059, 0, 0, 0, 0.056, 0, 0, 0, 0, 0.718, 0.080, 0.083]),
        60: np.array([0.055, 0, 0, 0, 0.055, 0, 0, 0, 0, 0.729, 0.082, 0.079]),
        75: np.array([0.054, 0, 0, 0, 0.054, 0, 0, 0, 0, 0.730, 0.082, 0.080]),
    },
    5: {
        0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        100: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    },
    6: {
        15: np.array([0.034, 0, 0.143, 0, 0.110, 0.128, 0, 0.164, 0.118, 0.207, 0.045, 0.046]),
        30: np.array([0.064, 0, 0.151, 0, 0.064, 0.148, 0, 0.128, 0.135, 0.215, 0.042, 0.042]),
        45: np.array([0.063, 0, 0.153, 0, 0.063, 0.151, 0, 0.134, 0.133, 0.223, 0.040, 0.036]),
        60: np.array([0.062, 0, 0.154, 0, 0.064, 0.151, 0, 0.133, 0.133, 0.219, 0.041, 0.039]),
        75: np.array([0.063, 0, 0.157, 0, 0.064, 0.154, 0, 0.129, 0.129, 0.220, 0.041, 0.037]),
    },
    7: {
        0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        100: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    },
}

# inlet_pwm转化为外循环比例
inlet_pwm_to_out_frac = np.array([[0, 0],
                                  [17, 17.8571],
                                  [30, 21.9178],
                                  [54, 60.8084],
                                  [66, 64.4404],
                                  [85, 68.9394],
                                  [100, 100]])

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
gas_liquid_mixing_ratio_table_path = './auto_get_data_from_pic/data/pkl/p_h_x_table_dict.pkl'
with open(gas_liquid_mixing_ratio_table_path, 'rb') as f:
    gas_liquid_table_dict = pickle.load(f)

p_h_x_log_p = gas_liquid_table_dict['log_p']
p_h_x_h = gas_liquid_table_dict['h']
p_h_x_x = gas_liquid_table_dict['x']

# 热熵温度表
p_h_t_table_path = './auto_get_data_from_pic/data/pkl/p_h_t_table_dict.pkl'
with open(p_h_t_table_path, 'rb') as f:
    p_h_t_table_dict = pickle.load(f)

p_h_t_log_p = p_h_t_table_dict['log_p']
p_h_t_h = p_h_t_table_dict['h']
p_h_t_t = p_h_t_table_dict['t']

'''流量相关查表'''
# 压缩机排量 单位cc ml^3

compressor_displacement = 33
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
