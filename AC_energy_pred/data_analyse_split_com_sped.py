import numpy as np
from torch.distributions.constraints import real_vector

from draw_pic import draw_pred

filename = "./data/data_anyls/output.npz"
data = np.load(filename)
pre_compressor_speed_r_list = []
real_compressor_speed_r_list = []
# 预测转速 饱和高压差 实际转速
for i in data:
    cur_dta = data[i]
    # 找到每个元素与其前一个元素不同的位置
    diff_indices = np.where(np.diff(np.sign(cur_dta[:,1])) != 0)[0] + 1
    # 将数据分成多个段
    slices = np.split(cur_dta, diff_indices)
    split_index = 0
    for row in slices:
        if row.shape[0] < 100:
            continue
        pre_com_sped = row[:,0]
        diff_hi_press = row[:, 1]
        real_com_sped = row[:, 2]

        series_list = [
        # [pred_t_r_1, refrigerant_mix_temp],
            [pre_com_sped, diff_hi_press, real_com_sped]
        ]
        series_name_list = [
        # ['pred_refrigerant_mix_temp', 'refrigerant_mix_temp'],
        ['pre_com_sped', 'diff_hi_press', 'real_com_sped'],
        ]
        result_pic_folder = './analyse_pic/'

        file_name = f'{i}_{split_index}'
        split_index = split_index + 1
        pic_name = file_name + '.png'
        # print(series_list)
        draw_pred(series_list, series_name_list, result_pic_folder, pic_name)
        pre_compressor_speed_r = np.corrcoef(pre_com_sped, diff_hi_press)[0][1]
        real_compressor_speed_r = np.corrcoef(real_com_sped, diff_hi_press)[0][1]
        print('file_name', file_name, 'pre_compressor_speed_r', pre_compressor_speed_r, 'real_compressor_speed_r',real_compressor_speed_r)
        if not np.isnan(pre_compressor_speed_r):
            pre_compressor_speed_r_list.append(pre_compressor_speed_r)
        if not np.isnan(real_compressor_speed_r):
            real_compressor_speed_r_list.append(real_compressor_speed_r)


print(np.mean(pre_compressor_speed_r_list))
print(np.mean(real_compressor_speed_r_list))

