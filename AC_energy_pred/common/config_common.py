import numpy as np

# 风道相关
all_wind_channel_temp_keys = [
    'TmRteOut_HvacFirstLeVentnAirT',  # 0 主驾驶吹面
    'TmRteOut_HvacFirstLeFlrAirT',  # 1 主驾驶吹脚
    'TmRteOut_HvacFirstRiVentnAirT',  # 2 副驾驶吹面
    'TmRteOut_HvacFirstRiFlrAirT',  # 3 副驾驶吹脚
    'TmRteOut_HvacSecLeVentnAirT',  # 4 二排吹面
    'TmRteOut_HvacSecLeFlrAirT',  # 5 后排吹脚
]

mix_door_control = {
    'mix_door_pwm_fl': ['TmRteOut_HvacFirstLeVentnAirT', 'TmRteOut_HvacFirstLeFlrAirT'],
    'mix_door_pwm_fr': ['TmRteOut_HvacFirstRiVentnAirT', 'TmRteOut_HvacFirstRiFlrAirT'],
    'mix_door_pwm_rl': ['TmRteOut_HvacSecLeVentnAirT', 'TmRteOut_HvacSecLeFlrAirT'],
}

# modena风门pwm相关
mix_door_pwm_list_modena = ['MixDor_L_PWM',  # 主驾驶温度风门pwm
                            'MixDor_R_PWM',  # 副驾驶温度风门pwm
                            'MIX_L_PID_Iout',  # 主驾驶温度风门pwm Iout
                            'MIX_R_PID_Iout',  # 副驾驶温度风门pwm Iout
                            'MIX_L_PID_Pout',  # 主驾驶温度风门pwm pout
                            'MIX_R_PID_Pout',  # 副驾驶温度风门pwm pout
                            ]
# lemans风门pwm相关
mix_door_pwm_list_lemans = ['MixDor_L_PWM',  # 主驾驶温度风门pwm
                            'MixDor_R_PWM',  # 副驾驶温度风门pwm
                            'MixDor_RL_PWM',  # 后排温度风门pwm
                            'MIX_L_PID_Iout',  # 主驾驶温度风门pwm Iout
                            'MIX_R_PID_Iout',  # 副驾驶温度风门pwm Iout
                            'MIX_RL_PID_Iout',  # 后排温度风门pwm Iout
                            'MIX_L_PID_Pout',  # 主驾驶温度风门pwm pout
                            'MIX_R_PID_Pout',  # 副驾驶温度风门pwm pout
                            'MIX_RL_PID_Pout',  # 后排温度风门pwm pout
                            ]
common_VAR_NAMES = [
    'COMM_VehicleSpd',  # 车速
    'TmRteOut_FrntIncarTRaw',  # 内温
    'TmRteOut_AmbTEstimd',  # 环境温度

    'TmRteOut_HvacFirstLeWindModSt',  # 乘员舱吹风模式
    'TmRteOut_FrntBlowerPwmReq',  # 100-空调占空比
    'DWT_DWT',  # Hvch出口目标水温
    'DHT_DHT',  # DHT,控制量

    'HP_Mode_Valve',  # 热泵模式 HP_Mode_AC的子模式

    'HMI_AutoLv_N',  # 前排自动等级 0:Low，1：Normal中，2：High
    'SubCANR_HvacCtrlMod',  # 是否开启auto模式 0关闭 1手动 2自动 可能没有 建议HMI_Blower_FL代替
    'HMI_Blower_FL',  # 手动前排0-8 63代表自动
    'HMI_Blower_RL',  # 后排挡位 手动自动都是 lemans 0关闭 1低 2高 modena 0关闭 63自动

    'HMI_Temp_FL',  # 主驾温度设定
    'HMI_Temp_FR',  # 主驾温度设定
    'HMI_Temp_RL',  # 后排温度设定

    'DVT_FL',  # 主驾目标进风温度
    'DVT_FR',  # 副驾目标进风温度
    'DVT_RL',  # 后排目标进风温度

    'Inlet_PWM',  # 内外循环DC电机占空比

]
lemans_other = [

    'SubCANR_HvacFirstLeWindModSt',  # 可能的前出风模式
    'SubCANR_HvacReLeWindModSt',  # 后排出风模式
    'Inlet_PWM_D_LIN',  # LIN电机占空比
    # 'Inlet_PWM',  # DC电机占空比
    # 'Act_BIn_Lin_MixX_ActPosLIN',  # 电机实际步数
    # 'Subsys_HvacFlapMot1PosnUDC',  # 电机实际电压

    'Outlet_SysN_FL',  # 前排备选 需要转化才是真实模式
    'Outlet_SysN_RL',  # 后排备选 需要转化才是真实模式

]

# 出风模式 热管理内部信号 转整车信号
mode_inner_to_can_first = {
    0: 0,
    1: 1,
    2: 3,
    3: 2,
    4: 6,
    5: 4,
    6: 7,
    7: 5,
}
mode_inner_to_can_second = {
    0: 0,
    1: 1,
    2: 3,
    3: 2,
    4: 3,
    5: 0,
    6: 3,
    7: 1,
}

'''阻力相关'''
W_A_modena = 2.43  # 迎风面积
W_A_lemans = 2.47  # 迎风面积
W_p = 1.2  # 空气密度
zero_temp_bias = -273.15  # 绝对0度
zero_air_density = 1.293  # 标准气压的0度的空气密度

# 训练时梯度/参数比例限制
gard_scale = 1

'''加热制冷模式相关参数'''
# 模式名字
cooling_mode_index = 2
heat_mode_index = 10

# 通道风温有效范围
max_channel_temp = 80

'''内外循环比例转化'''
# inlet_pwm转化为外循环比例modena
inlet_pwm_to_out_frac_modena = np.array([[0, 0],
                                         [17, 17.8571],
                                         [30, 21.9178],
                                         [54, 60.8084],
                                         [66, 64.4404],
                                         [85, 68.9394],
                                         [100, 100]])
inlet_gear_to_out_frac_modena = np.array([[0, 0],
                                         [1, 17.8571],
                                         [2, 21.9178],
                                         [3, 60.8084],
                                         [4, 64.4404],
                                         [5, 68.9394],
                                         [6, 100]])

# inlet_pwm转化为外循环比例lemans
inlet_gear_to_out_frac_lemans = np.array([[0, 3],
                                         [1, 17],
                                         [2, 20],
                                         [3, 40],
                                         [4, 51],
                                         [5, 64],
                                         [6, 90]])

inlet_pwm_to_out_frac_lemans = np.array([
    [0, 3],
    [40, 17],
    [100, 20],
])

# inlet_pwm_d_lin转化为外循环比例lemans
# inlet_pwm_d_lin_to_out_frac_lemans = np.array([
#     [60, 20],
#     [20, 40],
#     [14, 51],
#     [10, 64],
#     [0, 90],
# ])
inlet_pwm_d_lin_to_out_frac_lemans = np.array([
    [0, 90],
    [10, 64],
    [14, 51],
    [20, 40],
    [60, 20],
])
