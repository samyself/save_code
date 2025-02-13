import numpy as np
from torch.distributions.constraints import real_vector

from draw_pic import draw_pred

filename = "./data/data_anyls/output_all_input.npz"
data = np.load(filename)
# 压缩机排气温度、内冷温度、饱和高压、压缩机进气温度、饱和低压、目标饱和高压、目标过冷度、目标过热度、预测转速、饱和高压差、实际转速
for i in data:
    x = data[i]
    # 压缩机排气温度
    temp_p_h_2 = x[:, 0]*100
    # 内冷温度
    temp_p_h_5 = x[:, 1]*100
    # 饱和高压
    hi_pressure = x[:, 2]
    # 压缩机进气温度
    temp_p_h_1_cab_heating = x[:, 3]*100
    # 饱和低压
    lo_pressure = x[:, 4]
    # 目标饱和高压
    aim_hi_pressure = x[:, 5]
    # 目标饱和低压
    aim_lo_pressure = x[:, 6]
    # 目标过冷度
    sc_tar_mode_10 = x[:, 7]*100
    # 目标过热度
    sh_tar_mode_10 = x[:, 8]*100

    # 预测转速
    pre_com_sped = x[:, 9]
    # 饱和高压差
    hi_pressure_diff = x[:, 10]
    # 实际转速
    real_com_sped = x[:, 11]
    mask = abs(pre_com_sped - real_com_sped) > 1000
    for i in range( 5,len(mask)):
        if mask[i]:
            print('内冷温度',temp_p_h_5[i-5:i+5])


    # 绘图
    series_list = [
        # [pred_t_r_1, refrigerant_mix_temp],
        # [temp_p_h_2, temp_p_h_5, hi_pressure, temp_p_h_1_cab_heating, lo_pressure, aim_hi_pressure, aim_lo_pressure, sc_tar_mode_10, sh_tar_mode_10,pre_com_sped, hi_pressure_diff, real_com_sped],
        [temp_p_h_5, temp_p_h_1_cab_heating, pre_com_sped, hi_pressure_diff, real_com_sped],
    ]
    series_name_list = [
        # ['pred_refrigerant_mix_temp', 'refrigerant_mix_temp'],
        # ['temp_p_h_2', 'temp_p_h_5', 'hi_pressure', 'temp_p_h_1_cab_heating', 'lo_pressure', 'aim_hi_pressure', 'aim_lo_pressure', 'sc_tar_mode_10', 'sh_tar_mode_10','pre_com_sped', 'hi_pressure_diff', 'real_com_sped'],
        ['temp_p_h_5', 'temp_p_h_1_cab_heating', 'pre_com_sped', 'hi_pressure_diff', 'real_com_sped'],

    ]
    result_pic_folder = './analyse_pic/all5_com_sped_01/'

    file_name = f'{i}'
    pic_name = file_name + '.png'
    draw_pred(series_list, series_name_list, result_pic_folder, pic_name)
