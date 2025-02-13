# 导入需要用的包
import pandas as pd
import numpy as np
import os
from scipy.ndimage import median_filter
from PycharmProjects.pythonProject2.common_func import draw_pred

# 多阶中值滤波
def median_list_filter(x, sizes):
    filtered_signal = x.copy()
    for size in sizes:
        filtered_signal = median_filter(filtered_signal, size=size)
    return filtered_signal

# 求中值
def median(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    if n == 0:
        raise ValueError("The list is empty")

    mid = n // 2

    if n % 2 == 0:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2
    else:
        return sorted_lst[mid]

if __name__ == '__main__':
    csv_path = 'D:/Code/code/submit/bms_ai/PycharmProjects/data/filter_data/data_handle_new'
    save_path = 'D:/Code/code/submit/bms_ai/PycharmProjects/data/filter_data/data_handle_new_filter_test'
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)

    name_list = ['processed_HXMQSPZV2CRSGAAV6_1727273729499.0.csv',
                 'processed_LNBQ7XTEWNHGES1H8_1721797714348.0.csv',
                 'processed_LNBQB6XNSJJEAJEF5_1720885794884.0.csv',
                 'processed_HXMQ3GTZ0PXTEPZ90_1728281459726.0.csv'
                ]
    # 电流判断的窗口长度
    curr_wind_size = 10
    # 电压判断的窗口长度
    volt_wind_size = 3
    # 输入信息窗口长度
    data_seq_len = 300


    for name in name_list:
        df = pd.read_csv(os.path.join(csv_path, name))
        L = len(df['Curr'])
        if L < 300:
            continue
        avg_current_all = df['Curr'].rolling(window=L, min_periods=1).mean()
        avg_voltage_all = df['Volt'].rolling(window=L, min_periods=1).mean()
        avg_current_60 = df['Curr'].rolling(window=60, min_periods=1).mean()
        avg_voltage_60 = df['Volt'].rolling(window=60, min_periods=1).mean()
        avg_current_300 = df['Curr'].rolling(window=300, min_periods=1).mean()
        avg_voltage_300 = df['Volt'].rolling(window=300, min_periods=1).mean()
        avg_current_600 = df['Curr'].rolling(window=600, min_periods=1).mean()
        avg_voltage_600 = df['Volt'].rolling(window=600, min_periods=1).mean()


        old_curr = df['Curr'].copy()
        old_volt = df['Volt'].copy()

        # curr_10 = median_list_filter(df['Curr'], [10,5])
        # volt_10 = median_list_filter(df['Volt'], [10,5])
        df['Curr'] = median_list_filter(df['Curr'], [10])
        df['Volt'] = median_list_filter(df['Volt'], [10])

        # 起始位置
        start_idx = 0
        curr = df['Curr']
        req_curr = df['ReqCurr']
        volt = df['Volt']
        # 第i个数表示req_curr[i+1] - req_curr[i]
        req_curr_diff = np.diff(req_curr)

        for i in range(10,len(df) - curr_wind_size):
            if df['BmsSoc'][i]  < 70.0 or df['Curr'][i] < 5.0:
                start_idx = i+1
                continue
            else:
                pass
                if df['BmsSoc'][i]  < 80.0:
                    start_idx = max(i + 1-data_seq_len,start_idx)

                # 电流上升判断
                # if curr[i + 1] - curr[i] > 10.0 and median(curr[i:i + curr_wind_size]) - median(curr[i - curr_wind_size:i]) > 10.0 :
                if curr[i + 1] - curr[i] > 10.0:
                    if max(req_curr_diff[i - curr_wind_size: i]) >= 10.0 and median(
                            curr[i:i + curr_wind_size]) - median(curr[i - curr_wind_size:i]) > 10.0:
                        # 确定有上升沿，就往后延迟一个时间窗口
                        start_idx = max(i + data_seq_len, start_idx)
                    else:
                        pass
                # 电流下降判断
                elif curr[i + 1] - curr[i] < -10.0:
                    # 确定是下降沿
                    if min(req_curr_diff[i - curr_wind_size: i]) < -10.0 and median(
                            curr[i:i + curr_wind_size]) - median(curr[i - curr_wind_size:i]) < -10.0:
                        pass
                    # 噪声
                    else:
                        curr[i + 1] = curr[i].copy()

                # 电压下降沿判断
                if volt[i + 1] - volt[i] < -0.01:
                    # 第i个数表示curr[i+1] - curr[i]
                    curr_diff = np.diff(curr[i - volt_wind_size - 1: i])
                    # 确定是下降沿
                    if min(curr_diff) < -10.0 and median(volt[i:i + curr_wind_size]) - median(
                            volt[i - curr_wind_size:i]) < -0.01:
                        pass
                    else:
                        volt[i + 1] = volt[i].copy()


                # # 电流跳变点判断：
                # if (curr[i + 1:i+6] - curr[i] > 10.0).any() and (abs(curr[i -5:i] - curr[i]) < 10.0).any() and max(req_curr_diff[i - curr_wind_size : i]) >= 10.0:

                # 电流上升判断
                # if curr[i + 1] - curr[i] > 10.0 and median(curr[i:i + curr_wind_size]) - median(curr[i - curr_wind_size:i]) > 10.0 :
            # if abs(curr_10[i] - curr[i]) < 10.0:
            #     curr[i] = curr_10[i]
            # if abs(volt_10[i] - volt[i]) < 0.01:
            #     volt[i] = volt_10[i]
            #     (curr[i + 1:i+10] - curr[i] > 10.0).any() and (abs(curr[i + 2:i + 10] - curr[i + 1]) < 10.0).any() and
            #     if curr[i + 1] - curr[i] >  10.0 :
            #         if max(req_curr_diff[i - curr_wind_size : i]) >= 10.0 and median(curr[i:i + curr_wind_size]) - median(curr[i - curr_wind_size:i]) > 10.0:
            #             # 确定有上升沿，就往后延迟一个时间窗口
            #             start_idx = max(i + data_seq_len,start_idx)
            #         else:
            #             curr[i + 1] = curr[i]
            #             pass
                # 电流下降判断
                # elif curr[i + 1] - curr[i] < -10.0 :
                #     # 确定是下降沿
                #     #(curr[i + 1:i+10] - curr[i] < -10.0).any() and (abs(curr[i + 2:i + 10] - curr[i + 1]) < 10.0).any() and
                #     if min(req_curr_diff[i - curr_wind_size: i]) < -10.0 and median(curr[i:i + curr_wind_size]) - median(curr[i - curr_wind_size:i]) < -10.0:
                #         pass
                #     # 噪声
                #     else:
                #         curr[i + 1] = curr[i]
                #
                # # 电压下降沿判断
                # if volt[i + 1] - volt[i] < -0.01:
                #     # 第i个数表示curr[i+1] - curr[i]
                #     curr_diff = np.diff(curr[i - volt_wind_size - 1: i])
                #     # 确定是下降沿
                #     if min(curr_diff) < -10.0 and median(volt[i:i + curr_wind_size]) - median(volt[i - curr_wind_size:i]) < -0.01 :
                #         pass
                #     else:
                #         volt[i + 1] = volt[i]
                    # elif volt[i + 1] - volt[i] > 0.015:
                    #     # 确定是上升沿
                    #     if min(curr_diff[i - volt_wind_size: i]) > 5.0 and median(volt[i:i + curr_wind_size]) - median(volt[i - curr_wind_size:i]) > 0.015:
                    #         pass
                    #     else :
                    #         volt[i + 1] = volt[i]

        # 可视化
        series_list = [
            [old_curr, curr, req_curr],
            [old_volt, volt],

        ]
        series_name_list = [
            # ['pred_refrigerant_mix_temp', 'refrigerant_mix_temp'],
            ['old_curr', 'curr', 'req_curr'],
            ['old_volt', 'volt'],
        ]
        file_name = name.replace('.csv', '').split('/')[-1].split('\\')[-1]
        pic_name = file_name + '.png'
        # print(series_list)
        draw_pred(series_list, series_name_list, 'D:/Code/code/submit/bms_ai/PycharmProjects/data/filter_data/data_handle_new_filter_test', pic_name)
        df['Curr'] = curr
        df['Volt'] = volt
        df['OldCurr'] = old_curr
        df['OldVolt'] = old_volt
        if start_idx >= len(df)- data_seq_len - 100:
            continue
        split_df = df.iloc[start_idx:, :].reset_index(drop=True)
        split_df.to_csv(os.path.join(save_path, 'filter'+name), index=False)
