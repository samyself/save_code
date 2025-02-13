from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import os
from scipy.ndimage import median_filter
# from common_func import draw_pred


# def filter(x, sizes):
#     filtered_signal = x.copy()
#     for size in sizes:
#         filtered_signal = median_filter(filtered_signal, size=size)
#     return filtered_signal


# 数据预处理
def data_prepare1(T):
    # Volt = Check_Volt(T['Volt'], T['Curr'])
    # if (Volt != T['Volt']).any():
    #     print('起作用了',np.where(Volt != T['Volt']))

    T_origin = T.copy()

    condition1 = (T['BmsSoc'] >= 70.0)

    if condition1.any():
        idx_begin1 = np.where(condition1)[0][0]
    else:
        idx_begin1 = 0

    idx_begin = idx_begin1

    T = T.iloc[idx_begin:, :].reset_index(drop=True)
    T_origin = T_origin.iloc[idx_begin:, :].reset_index(drop=True)
    # 计算每个时间步下all数据的平均电流和平均电压

    # 判断有无小于0的值
    if ((T['BmsSoc']<=98.0) & (T['Curr'] <= 10.0)).any():
        return pd.DataFrame([])

    L = len(T['Curr'])
    if L < 300 or T['Curr'].values[0] < 100:
        return pd.DataFrame([])

    T['avg_current_all'] = T['Curr'].rolling(window=L, min_periods=1).mean()
    T['avg_voltage_all'] = T['Volt'].rolling(window=L, min_periods=1).mean()
    # T['avg_current_60'] = T['Curr'].rolling(window=60, min_periods=1).mean()
    # T['avg_voltage_60'] = T['Volt'].rolling(window=60, min_periods=1).mean()
    # T['avg_current_300'] = T['Curr'].rolling(window=300, min_periods=1).mean()
    # T['avg_voltage_300'] = T['Volt'].rolling(window=300, min_periods=1).mean()
    # T['avg_current_600'] = T['Curr'].rolling(window=600, min_periods=1).mean()
    # T['avg_voltage_600'] = T['Volt'].rolling(window=600, min_periods=1).mean()

    # 中值滤波
    # T['Curr'] = filter(T['Curr'],[10])
    # T['Volt'] = filter(T['Volt'],[10])

    # 计算Ah的变化量
    DtAh = (T['Ah'][1:].values - T['Ah'][:-1].values)

    T_each_all = pd.DataFrame({
        'time': T['time'][1:].values,
        'Volt': T['Volt'][1:].values,
        'Temp': T['Temp'][1:].values,
        'Curr': T['Curr'][1:].values,
        'BmsSoc': T['BmsSoc'][1:].values,
        'Ah': T['Ah'][1:].values,
        'DtAh': DtAh,
        # 'avg_current_60': T['avg_current_60'][1:].values,
        # 'avg_voltage_60': T['avg_voltage_60'][1:].values,
        # 'avg_current_300': T['avg_current_300'][1:].values,
        # 'avg_voltage_300': T['avg_voltage_300'][1:].values,
        # 'avg_current_600': T['avg_current_600'][1:].values,
        # 'avg_voltage_600': T['avg_voltage_600'][1:].values,
        'avg_current_all': T['avg_current_all'][1:].values,
        'avg_voltage_all': T['avg_voltage_all'][1:].values,
        'TrueSoc': T['TrueSoc'][1:].values,
    })
    return T_each_all


def merge_data(file_path_list, time_step, time_slip, time_skip, return_point):
    # 将给定df转为time_step * feature_num 的矩阵
    Xs, ys, info = [], [], []
    idx = 0
    for file_path in tqdm(file_path_list):
        # if idx >= 5000:
        #     break
        X_split = []
        y_split = []
        df_origin = pd.read_csv(file_path)
        df = df_origin.fillna(method='ffill').fillna(method='bfill')
        # 测试的时候，可以去掉这个部分
        df = data_prepare1(df)
        time_pred = 2

        if df.shape[0] < time_step + time_pred + 500:
            continue

        X = df[['Volt', 'Temp', 'Curr', 'Ah', 'DtAh',
                # 'avg_current_60', 'avg_voltage_60',
                # 'avg_current_300', 'avg_voltage_300',
                # 'avg_current_600', 'avg_voltage_600',
                'avg_current_all', 'avg_voltage_all', 'BmsSoc']].iloc[:-time_pred]
        X_curr = X['Curr'].values
        X_volt = X['Volt'].values

        X['Curr'] = X_curr
        X['Volt'] = X_volt

        y = df['TrueSoc'].values[time_pred:]
        L_length = len(X) - time_step - time_pred

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
                # avg_current_60 = x_all['avg_current_60'].values
                # avg_voltage_60 = x_all['avg_voltage_60'].values
                # avg_current_300 = x_all['avg_current_300'].values
                # avg_voltage_300 = x_all['avg_voltage_300'].values
                # avg_current_600 = x_all['avg_current_600'].values
                # avg_voltage_600 = x_all['avg_voltage_600'].values
                avg_current_all = x_all['avg_current_all'].values
                avg_voltage_all = x_all['avg_voltage_all'].values

                BmsSoc = x_all['BmsSoc'].values

                x_in = ([Volt] + [Curr] + [Ah] + [P_vxi] + [DtAh] +
                        [avg_current_all] + [avg_voltage_all] + [BmsSoc])

                # x_in = ([Volt] + [Curr] + [Ah] + [P_vxi] + [DtAh] + [avg_current_60] + [avg_voltage_60] +
                # [avg_current_300] + [avg_voltage_300] + [avg_current_600] + [avg_voltage_600] +
                # [avg_current_all] + [avg_voltage_all] + [BmsSoc])

                x_in = np.array(x_in).T

                y_out = np.array(y[i + time_step + 1])
                for i in range(1, time_pred):
                    y_out = np.append(y_out, y[i + time_step + 1])

                X_split.append(x_in)
                y_split.append(y_out)

        idx = idx + 1
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

