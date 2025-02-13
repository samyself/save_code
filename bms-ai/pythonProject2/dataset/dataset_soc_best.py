from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import os
from scipy.ndimage import median_filter
from ..common_func import draw_pred

def Check_Curr(curr,req_curr, window_size=10):
    # 第i个数表示req_curr[i+1] - req_curr[i]
    req_curr_diff = np.abs(np.diff(req_curr))
    for i in range(window_size,len(curr)):
        if curr[i] - curr[i-1] < -5.0:
            if np.max(req_curr_diff[i-10:i]) >= 8.0:
                pass
            else:
                curr[i] = curr[i-1]
        # elif curr[i] - curr[i-1] > 5.0:
        #         curr[i] = curr[i-1]
    return curr

def data_prepare1(T):

    condition1 = (T['BmsSoc'] >= 70.0)
    condition2 =(T['Curr'] <= 5.0)
    if condition1.any():
        idx_begin1 = np.where(condition1)[0][0]
    else:
        idx_begin1 = 0
    if condition2.any():
        idx_begin2 = np.where(condition2)[0][-1] + 1
    else:
        idx_begin2 = 0
    idx_begin = max(idx_begin1, idx_begin2)

    T = T.iloc[idx_begin:, :].reset_index(drop=True)

    # 计算每个时间步下all数据的平均电流和平均电压
    L = len(T['Curr'])
    if L < 300 :
        return pd.DataFrame([])
    avg_current_all = T['Curr'].rolling(window=L, min_periods=1).mean()
    avg_voltage_all = T['Volt'].rolling(window=L, min_periods=1).mean()
    # 预处理跳变的电流与电压
    Curr = Check_Curr(T['Curr'], T['ReqCurr'])
    if (Curr != T['Curr']).any():
        print('起作用了',np.where(Curr != T['Curr']))
    # 计算Ah的变化量
    DtAh = (T['Ah'][1:].values - T['Ah'][:-1].values)

    T_each_all = pd.DataFrame({
        'time': T['time'][1:].values,
        'TrueSoc': T['TrueSoc'][1:].values,
        'Volt': T['Volt'][1:].values,
        'Temp': T['Temp'][1:].values,
        'Curr': Curr[1:].values,
        'BmsSoc': T['BmsSoc'][1:].values,
        'Ah': T['Ah'][1:].values,
        'DtAh': DtAh,
        'avg_current_all': avg_current_all[1:].values,
        'avg_voltage_all': avg_voltage_all[1:].values
    })
    return T_each_all


def filter(x, sizes):
    filtered_signal = x.copy()
    for size in sizes:
        filtered_signal = median_filter(filtered_signal, size=size)
    return filtered_signal

def merge_data(file_path_list, time_step, time_slip, return_point):
    # 将给定df转为time_step * feature_num 的矩阵
    Xs, ys, info = [], [], []
    for file_path in tqdm(file_path_list):
        X_split =[]
        y_split = []
        df1 = pd.read_csv(file_path)
        df = data_prepare1(df1)

        time_pred = 3
        X = df[['Volt', 'Temp', 'Curr','Ah','DtAh',
                'avg_current_all', 'avg_voltage_all','BmsSoc']].iloc[:-time_pred]
        y = df['TrueSoc'].values[time_pred:]
        L_length = len(X) - time_step - time_pred
        for i in range(L_length):
            select_every_slip = range(i, i + time_step, time_slip)
            # 确保 select_every_slip 的范围不会超过序列的长度
            if max(select_every_slip) < L_length:
                x_all = X.iloc[select_every_slip]
                Volt = x_all['Volt'].values
                Curr = x_all['Curr'].values
                Curr = filter(Curr.reshape(-1, 1), [10, 5]).reshape(-1, )
                Volt = filter(Volt.reshape(-1, 1), [10, 5]).reshape(-1, )
                Ah = x_all['Ah'].values
                DtAh = x_all['DtAh'].values
                P_vxi = Volt * Curr
                # avg_current60 = x_all['avg_current60'].values
                # avg_voltage60 = x_all['avg_voltage60'].values
                # avg_current300 = x_all['avg_current300'].values
                # avg_voltage300 = x_all['avg_voltage300'].values
                # avg_current600 = x_all['avg_current600'].values
                # avg_voltage600 = x_all['avg_voltage600'].values
                avg_current_all = x_all['avg_current_all'].values
                avg_voltage_all = x_all['avg_voltage_all'].values

                BmsSoc = x_all['BmsSoc'].values

                # 将特征列表转换成一个数组
                x_in =  ([Volt] + [Curr] + [Ah] + [P_vxi] + [DtAh] +
                         [avg_current_all] + [avg_voltage_all] + [BmsSoc])
                x_in = np.array(x_in).T
                y_out = np.array(y[i + time_step + 1])
                for i in range(1,time_pred):
                    y_out = np.append(y_out, y[i + time_step + 1])

                X_split.append(x_in)
                y_split.append(y_out)

        if len(X_split) > 100:
            if return_point:
                Xs.extend(X_split)
                ys.extend(y_split)
            else:
                Xs.append(np.array(X_split))
                ys.append(np.array(y_split))
                file_name = os.path.basename(file_path)
                info.append([f'{file_name}'])

    # Xs = Xs)
    # ys = np.array(ys)
    return Xs, ys, info

# 制热模式数据 使用数据规则得到的ags开度风扇占空比 作为学习目标
class SOCBaseDataset(torch.utils.data.Dataset):
    def __init__(self, all_data_path, time_step=300, time_slip=1, return_point=True):
        self.all_data_path = all_data_path
        self.time_step = time_step
        self.time_slip = time_slip
        self.return_point = return_point

        self.all_x = []
        self.all_y = []
        self.info = []

        self.all_x , self.all_y, self.info = merge_data(self.all_data_path, self.time_step, self.time_slip, self.return_point)

    def __len__(self):
        return len(self.all_x)

    def __getitem__(self, item_index):
        x = self.all_x[item_index].astype('float32')
        y = self.all_y[item_index].astype('float32')
        return x, y