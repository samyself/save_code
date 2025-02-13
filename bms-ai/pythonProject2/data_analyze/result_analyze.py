import pandas as pd
import sys

sys.path.append('/home/work/meiyuheng_Project/bms_ai/PycharmProjects/pythonProject2')
import os

print(os.getcwd())
from PycharmProjects.pythonProject2.common_func import draw_pred, dictlist_to_listdict, create_histogram
import numpy as np
import json
from tqdm import tqdm
import threading


def analyze_csv(csv_path, save_folder):
    # 从CSV文件中加载DataFrame
    df = pd.read_csv(csv_path)

    # 创建掩码（mask）
    # mask1 = (df['Soc_real'] >= 80) & (df['Soc_real'] < 90) & (df['unhealthy_value'] < 1.0)
    # mask2 = (df['Soc_real'] >= 90) & (df['Soc_real'] <= 100) & (df['unhealthy_value'] < 0.0)
    # mask = mask1 | mask2
    # df = df.iloc[mask]
    # df = df[mask]
    analyze_info_list = [80.0, 90.0, 100.0]
    # 输出的统计信息
    output_info_list = []
    Soc_pr_erro_all = np.array([])
    for i in range(len(analyze_info_list) - 1):
        # if df['Curr'].values[0] < 100:
        #     break
        # 将不同分组分配到不同的文件
        save_folder_1 = f'{save_folder}_{analyze_info_list[i]}_{analyze_info_list[i + 1]}'
        # 创建一个新的DataFrame，包含特定列的数据
        # new_df = np.array(df[['Volt', 'Temp', 'Curr', 'Ah', 'P_vxi',
        #              'avg_current60', 'avg_voltage60', 'BmsSoc',
        #              'Soc_pred', 'Soc_real']])
        # 筛选出不同BMS大小的部分
        condition = (df['Soc_real'] >= analyze_info_list[i]) & (df['Soc_real'] < analyze_info_list[i + 1])
        if condition.sum() <= 0:
            break
        idx_begin = np.where(condition)[0][0]
        # print('idx_begin',idx_begin)
        idx_end = np.where(condition)[0][-1]
        # print('idx_end',idx_end)
        # print('df.shape',df.shape)
        df_new = df.iloc[idx_begin:idx_end + 1]
        # df_new = df_new.to_numpy()
        # print('df_new.shape',df_new.shape)
        # 预测误差
        Soc_pr_erro = np.abs(df_new['Soc_pred'] - df_new['Soc_real'])
        # BMS误差
        Soc_br_erro = np.abs(df_new['BmsSoc'] - df_new['Soc_real'])
        # # 可视化
        # series_list = [
        #     [df_new['Soc_pred'].to_numpy(), df_new['Soc_real'].to_numpy(), df_new['BmsSoc'].to_numpy()],
        #     [Soc_pr_erro.to_numpy(),Soc_br_erro.to_numpy()],
        #     [df_new['Volt'].to_numpy()],[df_new['Ah'].to_numpy()],[df_new['Curr'].to_numpy()],[df_new['P_vxi'].to_numpy()],
        #     [df_new['avg_current_all'].to_numpy()],[df_new['avg_voltage_all'].to_numpy()],
        # ]
        # series_name_list = [
        #     ['Soc_pred', 'Soc_real', 'BmsSoc'],
        #     ['Soc_pr_erro'],
        #     ['Volt'],['Ah'],['Curr'],['P_vxi'],
        #     ['avg_current_all'], ['avg_voltage_all'],
        # ]
        Soc_pr_mae = np.mean(Soc_pr_erro)
        soc_br_mae = np.mean(Soc_br_erro)
        Soc_pr_max = np.max(Soc_pr_erro)
        soc_br_max = np.max(Soc_br_erro)

        file_name = os.path.basename(csv_path).replace('.csv', '').split('/')[-1].split('\\')[-1]

        temp_dict = {}
        temp_dict['file_info'] = file_name + f"_{analyze_info_list[i]}_{analyze_info_list[i + 1]}"
        temp_dict['Soc_pr_mae'] = Soc_pr_mae
        temp_dict['soc_br_mae'] = soc_br_mae
        temp_dict['Soc_pr_max'] = Soc_pr_max
        temp_dict['soc_br_max'] = soc_br_max

        output_info_list.append(temp_dict)
        # # 可视化
        # pic_name = file_name + '.png'
        # draw_pred(series_list, series_name_list, save_folder_1, pic_name)
        Soc_pr_erro_all = np.concatenate([Soc_pr_erro_all, Soc_pr_erro], axis=0)

    return output_info_list, Soc_pr_erro_all


if __name__ == '__main__':
    intput_csv_folder = '../../data/csv_res/Soc_1203_lstm_v1'
    save_folder = '../../data/pic/Soc_1203_lstm_v1'
    json_path = '../../data/json_res/analyze_info_Soc_1203_lstm_v1.json'
    all_path_list = os.listdir(intput_csv_folder)
    print(all_path_list[0])

    all_info_list = []
    all_err = []
    with open(json_path, 'w') as f:
        print('len(all_path_list)', len(all_path_list))
        for i in tqdm(range(len(all_path_list))):
            path = all_path_list[i]
            if '.csv' in path:
                pass
            else:
                continue
            path = os.path.join(intput_csv_folder, path)
            output_dict, Soc_pr_erro = analyze_csv(path, save_folder)
            json.dump(output_dict, f, indent=4)
            f.write('\n')

            all_info_list.extend(output_dict)
            all_err.extend(Soc_pr_erro)

        # all_info_dict =  dictlist_to_listdict(all_info_list)
        # max_80_90_err = max(all_info_dict['Soc_pr_max'][range(0, len(all_info_dict['Soc_pr_max']),2)])
        # max_90_100_err = max(all_info_dict['Soc_pr_max'][range(1, len(all_info_dict['Soc_pr_max']), 2)])

        # print('80-90 max err: ', max_80_90_err)
        # print('90-100 max err: ', max_90_100_err)
        all_err_np = np.array(all_err)
        print('all max err: ', max(all_err))
        print('all err mae: ', np.mean(all_err_np))
        print('all err std: ', np.std(all_err_np))
        create_histogram(all_err)

