import os
import pickle
import sys
import numpy as np

project_folder = os.path.abspath('..')
sys.path.append(os.path.join(project_folder, 'common'))
from common import config_common

'''变量名字'''
# 出风口相关字段 modena
all_wind_outlet_temp_keys_modena = [
    'Tvent_Face_Drvr_Le',  # 0
    'Tvent_Face_Drvr_Ri',  # 1
    'Tvent_Foot_Drvr',  # 2
    'Tvent_Face_Pass_Le',  # 3
    'Tvent_Face_Pass_Ri',  # 4
    'Tvent_Foot_Pass',  # 5
    'Tvent_Face_Sec_Le',  # 6
    'Tvent_Face_Sec_Ri',  # 7
    'Tvent_Foot_Sec_Le',  # 8
    'Tvent_Foot_Sec_Ri',  # 9
    'Defog_wind_Le',  # 10
    'Defog_wind_Ri',  # 11
    'Defog_wind_Leside',  # 12
    'Defog_wind_Riside'  # 13
]
# 出风口相关字段 lemans
all_wind_outlet_temp_keys_lemans = [
    'Tvent_Face_Drvr_Le',  # 0
    'Tvent_Face_Drvr_Ri',  # 1
    'Tvent_Foot_Drvr',  # 2
    'Tvent_Face_Pass_Le',  # 3
    'Tvent_Face_Pass_Ri',  # 4
    'Tvent_Foot_Pass',  # 5
    'Tvent_Face_Sec_Le',  # 6
    'Tvent_Face_Sec_Ri',  # 7
    'Tvent_Foot_Sec_Le',  # 8
    'Tvent_Foot_Sec_Ri',  # 9
    'Defog_wind_Le',  # 10
    'Defog_wind_Ri',  # 11
    'Defog_wind_Leside',  # 12
    'Defog_wind_Riside',  # 13
    'Tvent_Face_B_Le',  # 14
    'Tvent_Face_B_Le',  # 15
]
all_wind_outlet_temp_keys_lemans_heihe = [

    'Tvent_Face_Drvr_Le',  # 0
    'Tvent_Face_Drvr_Ri',  # 1
    'Tvent_Foot_Drvr',  # 2
    'Tvent_Face_Pass_Le',  # 3
    'Tvent_Face_Pass_Ri',  # 4
    'Tvent_Foot_Pass',  # 5
    'Tvent_RearFace_Le',  # 6
    'Tvent_RearFace_Ri',  # 7
    'Tvent_RearFoot_Le',  # 8
    'Tvent_RearFoot_Ri',  # 9
    'T_Def_Centerleft',  # 10
    'T_Def_Centerright',  # 11
    'T_Def_Left',  # 12
    'T_Def_Right',  # 13
    'Tvent_B_Face_Le',  # 14
    'Tvent_B_Face_Ri',  # 15
]

# modena出风口数量
outlet_num_modena = 12

# lemans出风口数量
outlet_num_lemans = 12

base_VAR_NAMES = config_common.common_VAR_NAMES + [
    # 'COMM_VehicleSpd',  # 车速
    # 'TmRteOut_FrntIncarTRaw',  # 内温
    # 'TmRteOut_AmbTEstimd',  # 环境温度
    #
    # 'TmRteOut_HvacFirstLeWindModSt',  # 乘员舱吹风模式
    # 'TmRteOut_FrntBlowerPwmReq',  # 100-空调占空比

    'sTmSigIn_D_DrvrOcupcySts',  # 主驾驶占位信号
    'sTmSigIn_D_PassOcupcySts',  # 副驾驶占位信号
    'sTmSigIn_D_SecRowLeOcupcySts',  # 后排左占位信号
    'sTmSigIn_D_SecRowMidOcupcySts',  # 后排中占位信号
    'sTmSigIn_D_SecRowRiOcupcySts',  # 后排右占位信号

    'SEN_Solar_L',  # 左侧阳光
    'SEN_Solar_R',  # 右侧阳光

    # 'HMI_Temp_FL',  # 主驾温度设定
    # 'HMI_Temp_FR',  # 主驾温度设定
    # 'HMI_Temp_RL',  # 后排温度设定

    # 'DVT_FL',  # 主驾请求进风温度
    # 'DVT_FR',  # 副驾请求进风温度
    # 'DVT_RL',  # 后排请求进风温度

    # 'DWT_DWT',  # Hvch出口目标水温
    # 'DHT_DHT',  # DHT,控制量
    #
    # 'HP_Mode_Valve',  # 热泵模式 HP_Mode_AC的子模式

    'HPowDVT_CabinZero_FL2',  # 主驾驶 初始内温 - setpoint
    'HPowDVT_CabinZero_FR2',  # 副驾驶 初始内温 - setpoint
    'HPowDVT_CabinZero_RL2',  # 后排 初始内温 - setpoint

    'HPowDVT_SteadyDVT_FL',  # 主驾驶稳态请求风温
    'HPowDVT_SteadyDVT_FR',  # 副驾驶稳态请求风温
    'HPowDVT_SteadyDVT_RL',  # 后排稳态请求风温

    'HPowDVT_PowDVT_FL_Ref2',  # 主驾驶 请求风温1（稳态） +  请求风温2（瞬态）
    'HPowDVT_PowDVT_FR_Ref2',  # 副驾驶 请求风温1（稳态） +  请求风温2（瞬态）
    'HPowDVT_PowDVT_RL_Ref2',  # 后排 请求风温1（稳态） +  请求风温2（瞬态）

    'HPowBlw_SteadyPWM_FL',  # 主驾驶稳态请求风量
    'HPowBlw_SteadyPWM_FR',  # 副驾驶稳态请求风量
    'HPowBlw_SteadyPWM_RL',  # 后排稳态请求风量

    'HPowBlw_PowPWM_FL2',  # 主驾驶 请求风量1（稳态） +  请求风量2（瞬态）
    'HPowBlw_PowPWM_FR2',  # 副驾驶 请求风量1（稳态） +  请求风量2（瞬态）
    'HPowBlw_PowPWM_RL2',  # 后排 请求风量1（稳态） +  请求风量2（瞬态）

    'HPowBlw_PowPWM_Front2',  # 最终 请求风量1（稳态） +  请求风量2（瞬态）
    'Blower_Fr_PWM',  # 最终请求pwm 即下一时刻响应的pwm

    # 'HMI_AutoLv_N',  # 前排自动等级
    # 'SubCANR_HvacCtrlMod',  # 是否开启auto模式 0关闭 1手动 2自动 可能没有 建议HMI_Blower_FL代替
    # 'HMI_Blower_FL',  # 手动前排0-8 63代表自动
    # 'HMI_Blower_RL',  # 后排挡位 手动自动都是 lemans 0关闭 1低 2高 modena 0关闭 63自动

    # 'MixDor_L_PWM',  # 主驾驶风门pwm
    # 'MixDor_R_PWM',  # 副驾驶风门pwm
    # 'MixDor_RL_PWM',  # 后排风门pwm

] + config_common.all_wind_channel_temp_keys

VAR_NAMES_modena = base_VAR_NAMES + config_common.mix_door_pwm_list_modena + all_wind_outlet_temp_keys_modena
VAR_NAMES_lemans = base_VAR_NAMES + config_common.mix_door_pwm_list_lemans + all_wind_outlet_temp_keys_lemans + \
                   all_wind_outlet_temp_keys_lemans_heihe + config_common.lemans_other

# HMI_Blower_FL什么时候代表自动
hmi_auto_pad = 63
# SubCANR_HvacCtrlMod什么时候代表自动
auto_mode = 2

# 出风口跟哪些字段有关modena
all_wind_temp_outlet_list_modena = [
    'temp_face_drvr_le',  # 0
    'temp_face_drvr_ri',  # 1
    'temp_foot_drvr',  # 2
    'temp_face_pass_le',  # 3
    'temp_face_pass_ri',  # 4
    'temp_foot_pass',  # 5
    'temp_face_sec',  # 6
    'temp_foot_sec_le',  # 7
    'temp_foot_sec_ri',  # 8
    'defog',  # 9
    'defog_side_le',  # 10
    'defog_side_ri',  # 11
]
# 风道风量转换表modena
vol_rate_outlet_to_channel_dict_modena = {
    'TmRteOut_HvacFirstLeVentnAirT': [0, 1],  # 0 主驾驶吹面
    'TmRteOut_HvacFirstLeFlrAirT': [2],  # 1 主驾驶吹脚
    'TmRteOut_HvacFirstRiVentnAirT': [3, 4],  # 2 副驾驶吹面
    'TmRteOut_HvacFirstRiFlrAirT': [5],  # 3 副驾驶吹脚
    'TmRteOut_HvacSecLeVentnAirT': [6],  # 4 二排吹面
    'TmRteOut_HvacSecLeFlrAirT': [7, 8],  # 5 后排吹脚
}
all_wind_temp_outlet_dict_modena = {
    'temp_face_drvr_le': ['Tvent_Face_Drvr_Le'],  # 0
    'temp_face_drvr_ri': ['Tvent_Face_Drvr_Ri'],  # 1
    'temp_foot_drvr': ['Tvent_Foot_Drvr'],  # 2
    'temp_face_pass_le': ['Tvent_Face_Pass_Le'],  # 3
    'temp_face_pass_ri': ['Tvent_Face_Pass_Ri'],  # 4
    'temp_foot_pass': ['Tvent_Foot_Pass'],  # 5
    'temp_face_sec': ['Tvent_Face_Sec_Le', 'Tvent_Face_Sec_Ri'],  # 6
    'temp_foot_sec_le': ['Tvent_Foot_Sec_Le'],  # 7
    'temp_foot_sec_ri': ['Tvent_Foot_Sec_Ri'],  # 8
    'defog': ['Defog_wind_Le', 'Defog_wind_Ri'],  # 9
    'defog_side_le': ['Defog_wind_Leside'],  # 10
    'defog_side_ri': ['Defog_wind_Riside'],  # 11
}

# 出风口跟哪些字段有关lemans
all_wind_temp_outlet_list_lemans = [
    'temp_face_drvr',  # 0
    'temp_foot_drvr',  # 1
    'temp_face_pass',  # 2
    'temp_foot_pass',  # 3
    'temp_face_sec',  # 4
    'temp_foot_sec_le',  # 5
    'temp_foot_sec_ri',  # 6
    'defog',  # 7
    'defog_side_le',  # 8
    'defog_side_ri',  # 9
    'temp_face_b_le',  # 10
    'temp_face_b_ri',  # 11
]
# 风道风量转换表lemans
vol_rate_outlet_to_channel_dict_lemans = {
    'TmRteOut_HvacFirstLeVentnAirT': [0],  # 0 主驾驶吹面
    'TmRteOut_HvacFirstLeFlrAirT': [1],  # 1 主驾驶吹脚
    'TmRteOut_HvacFirstRiVentnAirT': [2],  # 2 副驾驶吹面
    'TmRteOut_HvacFirstRiFlrAirT': [3],  # 3 副驾驶吹脚
    'TmRteOut_HvacSecLeVentnAirT': [4, 10, 11],  # 4 二排吹面
    'TmRteOut_HvacSecLeFlrAirT': [5, 6],  # 5 后排吹脚
}
all_wind_temp_outlet_dict_lemans = {
    'temp_face_drvr': ['Tvent_Face_Drvr_Le', 'Tvent_Face_Drvr_Ri'],  # 0
    'temp_foot_drvr': ['Tvent_Foot_Drvr'],  # 1
    'temp_face_pass': ['Tvent_Face_Pass_Le', 'Tvent_Face_Pass_Ri'],  # 2
    'temp_foot_pass': ['Tvent_Foot_Pass'],  # 3
    'temp_face_sec': ['Tvent_Face_Sec_Le', 'Tvent_Face_Sec_Ri'],  # 4
    'temp_foot_sec_le': ['Tvent_Foot_Sec_Le'],  # 5
    'temp_foot_sec_ri': ['Tvent_Foot_Sec_Ri'],  # 6
    'defog': ['Defog_wind_Le', 'Defog_wind_Ri'],  # 7
    'defog_side_le': ['Defog_wind_Leside'],  # 8
    'defog_side_ri': ['Defog_wind_Riside'],  # 9
    'temp_face_b_le': ['Tvent_Face_B_Le'],  # 10
    'temp_face_b_ri': ['Tvent_Face_B_Le'],  # 11
}
all_wind_temp_outlet_dict_lemans_heihe = {
    'temp_face_drvr': ['Tvent_Face_Drvr_Le', 'Tvent_Face_Drvr_Ri'],  # 0
    'temp_foot_drvr': ['Tvent_Foot_Drvr'],  # 1
    'temp_face_pass': ['Tvent_Face_Pass_Le', 'Tvent_Face_Pass_Ri'],  # 2
    'temp_foot_pass': ['Tvent_Foot_Pass'],  # 3
    'temp_face_sec': ['Tvent_RearFace_Le', 'Tvent_RearFace_Ri'],  # 4
    'temp_foot_sec_le': ['Tvent_RearFoot_Le'],  # 5
    'temp_foot_sec_ri': ['Tvent_RearFoot_Ri'],  # 6
    'defog': ['T_Def_Centerleft', 'T_Def_Centerright'],  # 7
    'defog_side_le': ['T_Def_Left'],  # 8
    'defog_side_ri': ['T_Def_Right'],  # 9
    'temp_face_b_le': ['Tvent_B_Face_Le'],  # 10
    'temp_face_b_ri': ['Tvent_B_Face_Ri'],  # 11
}

'''训练相关'''
lr = 1e-1
train_batch_size = 1024
train_epoch = 10
eval_batch_size = 1
gard_scale = 1
min_split_len = 100
base_dataset_folder = './data/datset_pkl/'

'''转化所需的表'''
# 空调乘员舱 占空比转化为总风量  lemans 前模式 后模式 前等级 后等级 是否自动模式
pwm_to_vol_lemans = {
    (0, 0, 0, 0, 0): {
        20: 0,
        70: 0,
    },
    (1, 1, 1, 2, 1): {
        20: 70.4,
        30: 138.8,
        40: 220.6,
        50: 306.8,
        60: 390.7,
        65: 432.7,
        70: 475,
        75: 475.4
    },
    (1, 0, 1, 0, 1): {
        20: 81.4,
        40: 204.2,
        60: 326.7,
        65: 358,
    },
    (2, 2, 1, 2, 1): {
        20: 54,
        30: 105,
        40: 156,
        50: 217.1,
        60: 278.2,
        65: 308,
        70: 339.5
    },
    (3, 3, 1, 2, 1): {
        20: 54,
        30: 105,
        40: 156,
        50: 217.1,
        60: 278.2,
        65: 308,
        70: 339.5
    },
    (6, 3, 1, 2, 1): {
        20: 53.4,
        30: 103.6,
        40: 153.7,
        50: 211.5,
        60: 269.2,
        65: 299,
    },
    (7, 3, 1, 2, 1): {
        20: 57,
        30: 113.8,
        40: 170.5,
        50: 239.8,
        60: 309,
        65: 347.3,
        70: 385.4,
    },
    (4, 0, 5, 0, 0): {
        20: 57,
        30: 113.8,
        40: 170.5,
        50: 239.8,
        60: 309,
        65: 347.3,
        70: 385.4,
    },
}
# 若无对应模式的默认总风量查表
default_pwm_to_vol_lemans = {
    20: 54,
    30: 105,
    40: 156,
    50: 217.1,
    60: 278.2,
    65: 308,
    70: 339.5
}

# 风量比例 lemans todo 当前是固定的 和pwm无关
mode_to_vol_rate_lemans = {
    (0, 0, 0, 0, 0): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) / 100,
    (1, 1, 1, 2, 1): np.array([38.7, 0, 35.3, 0, 13.6, 0, 0, 0, 0, 0, 6.4, 6]) / 100,
    (1, 0, 1, 0, 1): np.array([51.1, 0, 48.9, 0, 0, 0, 0, 0, 0, 0, 0, 0]) / 100,
    (2, 2, 1, 2, 1): np.array([6.162, 14.976, 4.134, 15.99, 0., 19.5, 17.238, 10, 6.0, 6.0, 0, 0]) / 100,
    (3, 3, 1, 2, 1): np.array([25.8, 7.6, 24, 8.5, 7.8, 10.1, 9.4, 0, 0, 0, 3.5, 3.3]) / 100,
    (6, 3, 1, 2, 1): np.array([4.3, 10.9, 2.8, 12.2, 11.5, 15, 13.8, 8.7, 5.5, 5.9, 4.8, 4.6]) / 100,
    (7, 3, 1, 2, 1): np.array([17.2, 8.3, 14.3, 9.4, 9, 11.2, 10.4, 5.2, 3.4, 3.7, 4, 3.9]) / 100,
    (4, 0, 5, 0, 0): np.array([0, 0, 0, 0, 0, 0, 0, 84.8, 7.5, 7.7, 0, 0]) / 100,
}
# 理想风量比例 lemans 自动模式 挡位 前中 后高 是否自动模式
ideal_mode_to_vol_rate_lemans = {
    (0, 0, 0, 0, 0): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) / 100,  # 这个一旦是前排为0 后排无论什么都按照这个
    (1, 1, 1, 2, 1): np.array([35, 0, 35, 0, 15, 0, 0, 0, 0, 0, 7.5, 7.5]) / 100,
    (2, 2, 1, 2, 1): np.array([5, 17.5, 5, 17.5, 0, 17.5, 17.5, 15, 2.5, 2.5, 0, 0]) / 100,
    (3, 3, 1, 2, 1): np.array([25, 12, 25, 12, 5, 8, 8, 0, 0, 0, 2.5, 2.5]) / 100,
    (4, 0, 5, 0, 0): np.array([5, 0, 5, 0, 0, 0, 0, 70, 10, 10, 0, 0]) / 100,  # 这个一旦是前排为4 后排无论什么都按照这个
    (5, 1, 1, 2, 1): np.array([22, 0, 22, 0, 11, 0, 0, 35, 5, 5, 0, 0]) / 100,
    (6, 3, 1, 2, 1): np.array([5, 12.5, 5, 12.5, 0, 12.5, 12.5, 30, 5, 5, 0, 0]) / 100,
    (7, 3, 1, 2, 1): np.array([15, 10.5, 15, 10.5, 7.5, 7, 7, 15, 2.5, 2.5, 3.75, 3.75]) / 100,

}

# 空调乘员舱 占空比转化为总风量 modena
pwm_to_vol_modena = {
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

# 风量比例 modena
mode_to_vol_rate_modena = {
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
