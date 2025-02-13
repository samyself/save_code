from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import os
from scipy.ndimage import median_filter
from ..common_func import draw_pred
from scipy.ndimage import median_filter


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


# 前处理
def pre_filter(df):
    # 电流判断的窗口长度
    curr_wind_size = 10
    # 电压判断的窗口长度
    volt_wind_size = 3
    # 输入信息窗口长度
    data_seq_len = 300

    L = len(df['Curr'])
    if L < 300:
        return pd.DataFrame()

    old_df = df.copy()
    condition1 = (old_df['BmsSoc'] >= 70.0)
    # 电流过低
    condition2 = (old_df['Curr'] <= 5.0)

    if condition1.any():
        idx_begin1 = np.where(condition1)[0][0]
    else:
        idx_begin1 = 0
    if condition2.any():
        idx_begin2 = np.where(condition2)[0][0]
    else:
        idx_begin2 = 0
    idx_begin = max(idx_begin1, idx_begin2)

    avg_current_all = df['Curr'].rolling(window=L, min_periods=1).mean()
    avg_voltage_all = df['Volt'].rolling(window=L, min_periods=1).mean()
    avg_current_60 = df['Curr'].rolling(window=60, min_periods=1).mean()
    avg_voltage_60 = df['Volt'].rolling(window=60, min_periods=1).mean()
    avg_current_300 = df['Curr'].rolling(window=300, min_periods=1).mean()
    avg_voltage_300 = df['Volt'].rolling(window=300, min_periods=1).mean()
    avg_current_600 = df['Curr'].rolling(window=600, min_periods=1).mean()
    avg_voltage_600 = df['Volt'].rolling(window=600, min_periods=1).mean()


    df['Curr'] = median_list_filter(df['Curr'], [10])
    df['Volt'] = median_list_filter(df['Volt'], [10])
    # 起始位置
    start_idx = 0
    curr = df['Curr'].copy()
    req_curr = df['ReqCurr'].copy()
    volt = df['Volt'].copy()
    # 第i个数表示req_curr[i+1] - req_curr[i]
    req_curr_diff = np.diff(req_curr)

    # 预处理电流、电压、起始位置
    for i in range(len(df) - curr_wind_size):
        if df['BmsSoc'][i] < 70.0 or df['Curr'][i] < 5.0:
            start_idx = i + 1
            continue
        else:
            pass
            if df['BmsSoc'][i] < 80.0:
                start_idx = max(i + 1 - data_seq_len, start_idx)

            # 电流上升判断
            # if curr[i + 1] - curr[i] > 10.0 and median(curr[i:i + curr_wind_size]) - median(curr[i - curr_wind_size:i]) > 10.0 :
            if curr[i + 1] - curr[i] > 10.0:
                if max(req_curr_diff[i - curr_wind_size: i]) >= 10.0 and median(curr[i:i + curr_wind_size]) - median(
                        curr[i - curr_wind_size:i]) > 10.0:
                    # 确定有上升沿，就往后延迟一个时间窗口
                    start_idx = max(i + data_seq_len, start_idx)
                else:
                    pass
            # 电流下降判断
            elif curr[i + 1] - curr[i] < -10.0:
                # 确定是下降沿
                if min(req_curr_diff[i - curr_wind_size: i]) < -10.0 and median(curr[i:i + curr_wind_size]) - median(
                        curr[i - curr_wind_size:i]) < -10.0:
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
                    volt[i + 1] = volt[i]

    if start_idx >= len(df) - data_seq_len - 100:
        return pd.DataFrame()

    # 计算Ah的变化量
    DtAh = (df['Ah'][1:].values - df['Ah'][:-1].values)
    T_each_all = pd.DataFrame({
        'time': df['time'][1:].values,
        'Volt': df['Volt'][1:].values,
        'Temp': df['Temp'][1:].values,
        'Curr': df['Curr'][1:].values,
        'BmsSoc': df['BmsSoc'][1:].values,
        'Ah': df['Ah'][1:].values,
        'DtAh': DtAh,
        'avg_current_60': avg_current_60[1:].values,
        'avg_voltage_60': avg_voltage_60[1:].values,
        'avg_current_300': avg_current_300[1:].values,
        'avg_voltage_300': avg_voltage_300[1:].values,
        'avg_current_600': avg_current_600[1:].values,
        'avg_voltage_600': avg_voltage_600[1:].values,
        'avg_current_all': avg_current_all[1:].values,
        'avg_voltage_all': avg_voltage_all[1:].values,
        'TrueSoc': df['TrueSoc'][1:].values,
    })

    split_df = T_each_all.iloc[start_idx:, :].reset_index(drop=True)
    return split_df


def merge_data(file_path_list, time_step, time_slip, time_skip, return_point):
    # 将给定df转为time_step * feature_num 的矩阵
    Xs, ys, info = [], [], []
    for file_path in tqdm(file_path_list):
        X_split = []
        y_split = []
        df_origin = pd.read_csv(file_path)
        # # 前向填充
        # ffill_df = df_origin.ffill()
        # # 后向填充
        # bfill_df = df_origin.bfill()
        # # 组合填充
        # df1 = ffill_df.combine_first(bfill_df)
        time_pred = 3
        df = pre_filter(df_origin)
        if df.empty or len(df) < 400 or len(df.shape) != 2:
            continue
        if df == pd.DataFrame():
            print('1')

        X = df[['Volt', 'Temp', 'Curr', 'Ah', 'DtAh',
                'avg_current_60', 'avg_voltage_60',
                'avg_current_300', 'avg_voltage_300',
                'avg_current_600', 'avg_voltage_600',
                'avg_current_all', 'avg_voltage_all', 'BmsSoc']].iloc[:-time_pred]
        # X_curr = X['Curr'].values
        # X_volt = X['Volt'].values
        #
        # X['Curr'] = X_curr
        # X['Volt'] = X_volt

        y = df['TrueSoc'].values[time_pred:]
        L_length = len(X) - time_step - time_pred

        # if L_length < 100:
        #     continue

        for i in range(0, L_length, time_skip):
            select_every_slip = range(i, i + time_step, time_slip)
            # 确保 select_every_slip 的范围不会超过序列的长度
            if max(select_every_slip) < L_length:
                x_all = X.iloc[select_every_slip]
                Volt = x_all['Volt'].values
                Curr = x_all['Curr'].values
                Ah = x_all['Ah'].values
                P_vxi = Volt * Curr
                DtAh = x_all['DtAh'].values
                avg_current_60 = x_all['avg_current_60'].values
                avg_voltage_60 = x_all['avg_voltage_60'].values
                avg_current_300 = x_all['avg_current_300'].values
                avg_voltage_300 = x_all['avg_voltage_300'].values
                avg_current_600 = x_all['avg_current_600'].values
                avg_voltage_600 = x_all['avg_voltage_600'].values
                avg_current_all = x_all['avg_current_all'].values
                avg_voltage_all = x_all['avg_voltage_all'].values

                BmsSoc = x_all['BmsSoc'].values

                # 将特征列表转换成一个数组
                # x_in =  ([Volt] + [Curr] + [Ah] + [P_vxi] +
                #          [avg_current_60] + [avg_voltage_60] +
                #          [avg_current_300] + [avg_voltage_300] +
                #          [avg_current_600] + [avg_voltage_600] +
                #          [avg_current_all] + [avg_voltage_all] + [BmsSoc])

                x_in = ([Volt] + [Curr] + [Ah] + [P_vxi] + [DtAh] +
                        [avg_current_all] + [avg_voltage_all] + [BmsSoc])
                x_in = np.array(x_in).T
                # x_mean = np.array([3.463, 81.054740, 124.407585, 281.342,
                #                    3.469, 97.281070, 3.469, 97.281070,
                #                    3.469, 97.281070, 3.469, 97.281070, 0.0])
                # x_std = np.array([0.022, 32.090257, 22.388481, 112.728,
                #                   0.019, 32.608585, 0.019, 32.608585,
                #                  0.019, 32.608585,0.019, 32.60858, 1.0])
                x_mean = np.array([3.463, 81.054740, 124.407585, 281.342, 0.0,
                                   3.469, 97.281070, 0.0])
                x_std = np.array([0.022, 32.090257, 22.388481, 112.728, 1.0,
                                  0.019, 32.608585, 1.0])

                # x_in = z_score(x_in, x_mean, x_std)
                y_out = np.array(y[i + time_step + 1])
                for i in range(1, time_pred):
                    y_out = np.append(y_out, y[i + time_step + 1])

                X_split.append(x_in)
                y_split.append(y_out)
        if len(X_split) <= 0:
            continue

        if return_point:
            Xs.extend(X_split)
            ys.extend(y_split)
        else:
            Xs.append(np.array(X_split))
            ys.append(np.array(y_split))
            file_name = os.path.basename(file_path)
            info.append([f'{file_name}'])

    return Xs, ys, info


# 制热模式数据 使用数据规则得到的ags开度风扇占空比 作为学习目标
class SOCBaseDataset(torch.utils.data.Dataset):
    def __init__(self, all_data_path, time_step=300, time_slip=1, time_skip=1, return_point=True):
        self.all_data_path = all_data_path
        # 用到的时间窗口长度
        self.time_step = time_step
        # 时间窗口长度中采用的步长
        self.time_slip = time_slip
        self.return_point = return_point
        # 窗口滑动步长
        self.time_skip = time_skip

        self.all_x = []
        self.all_y = []
        self.info = []

        self.all_x, self.all_y, self.info = merge_data(self.all_data_path, self.time_step, self.time_slip,
                                                       self.time_skip, self.return_point)

    def __len__(self):
        return len(self.all_x)

    def __getitem__(self, item_index):
        x = self.all_x[item_index].astype('float32')
        y = self.all_y[item_index].astype('float32')
        return x, y