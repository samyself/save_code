from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import os
from scipy.ndimage import median_filter
from ..common_func import draw_pred

def filter(x, sizes):
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

# 检测最后一个上升沿的位置
def up_check(target_list, set_value, window_size=10):
    last_up = 0
    for i in range(window_size,len(target_list)-window_size):
        if target_list[i+1] - target_list[i] > set_value and median(target_list[i:i+window_size]) - median(target_list[i - window_size:i]) > set_value:
            last_up = i
    return last_up


def Check_Curr(curr,req_curr, window_size=10):
    # 第i个数表示req_curr[i+1] - req_curr[i]
    req_curr_diff = np.abs(np.diff(req_curr))
    for i in range(window_size,len(curr)):
        if curr[i] - curr[i-1] < -5.0:
            if np.max(req_curr_diff[i-10:i]) >= 8.0:
                pass
            else:
                curr[i] = curr[i-1]
    return curr

def Check_Volt(volt,curr, window_size=3):
    # 第i个数表示curr[i+1] - curr[i]
    curr_diff = np.abs(np.diff(curr))
    for i in range(window_size,len(volt)):
        if volt[i] - volt[i-1] < -0.01:
            if np.min(curr_diff[i-window_size:i]) <= -10.0:
                pass
            else:
                curr[i] = curr[i-1]
        elif curr[i] - curr[i-1] > 0.015:
            if np.max(curr_diff[i - window_size:i]) >= 5.0:
                pass
            else:
                curr[i] = curr[i-1]
    return curr

# 数据预处理
def data_prepare1(T):
    condition1 = (T['BmsSoc'] >= 70.0)
    # 电流过低
    condition2 =(T['Curr'] <= 5.0)
    if condition1.any():
        idx_begin1 = np.where(condition1)[0][0]
    else:
        idx_begin1 = 0
    if condition2.any():
        idx_begin2 = np.where(condition2)[0][-1] + 1
    else:
        idx_begin2 = 0
    idx_begin = max([idx_begin1, idx_begin2])

    T = T.iloc[idx_begin:, :].reset_index(drop=True)

    # 计算每个时间步下all数据的平均电流和平均电压
    L = len(T['Curr'])
    if L < 300 :
        return pd.DataFrame([])
    avg_current_all = T['Curr'].rolling(window=L, min_periods=1).mean()
    avg_voltage_all = T['Volt'].rolling(window=L, min_periods=1).mean()
    avg_current_60 = T['Curr'].rolling(window=60, min_periods=1).mean()
    avg_voltage_60 = T['Volt'].rolling(window=60, min_periods=1).mean()
    avg_current_300 = T['Curr'].rolling(window=300, min_periods=1).mean()
    avg_voltage_300 = T['Volt'].rolling(window=300, min_periods=1).mean()
    avg_current_600 = T['Curr'].rolling(window=600, min_periods=1).mean()
    avg_voltage_600 = T['Volt'].rolling(window=600, min_periods=1).mean()

    # 预处理跳变的电流与电压
    Curr = Check_Curr(T['Curr'], T['ReqCurr'])
    if (Curr != T['Curr']).any():
        print('起作用了',np.where(Curr != T['Curr']))
    Volt = Check_Volt(T['Volt'], T['Curr'])
    if (Volt != T['Volt']).any():
        print('起作用了',np.where(Volt != T['Volt']))

    # 计算Ah的变化量
    DtAh = (T['Ah'][1:].values - T['Ah'][:-1].values)

    T_each_all = pd.DataFrame({
        'time': T['time'][:-1].values,
        'Volt': Volt[:-1].values,
        'Temp': T['Temp'][:-1].values,
        'Curr': Curr[:-1].values,
        'BmsSoc': T['BmsSoc'][:-1].values,
        'Ah': T['Ah'][:-1].values,
        'avg_current_60': avg_current_60[:-1].values,
        'avg_voltage_60': avg_voltage_60[:-1].values,
        'avg_current_300': avg_current_300[:-1].values,
        'avg_voltage_300': avg_voltage_300[:-1].values,
        'avg_current_600': avg_current_600[:-1].values,
        'avg_voltage_600': avg_voltage_600[:-1].values,
        'avg_current_all': avg_current_all[:-1].values,
        'avg_voltage_all': avg_voltage_all[:-1].values,
        'TrueSoc': T['TrueSoc'][1:].values,
    })
    return T_each_all


def merge_data(file_path_list, time_step, time_slip, time_skip, return_point):
    # 将给定df转为time_step * feature_num 的矩阵
    Xs, ys, info = [], [], []
    for file_path in tqdm(file_path_list):
        X_split =[]
        y_split = []
        df_origin = pd.read_csv(file_path)
        # 前向填充
        ffill_df = df_origin.ffill()
        # 后向填充
        bfill_df = df_origin.bfill()
        # 组合填充
        df = ffill_df.combine_first(bfill_df)
        time_pred = 2
        df = data_prepare1(df)

        X = df[['Volt', 'Temp', 'Curr','Ah',
                'avg_current_60', 'avg_voltage_60',
                'avg_current_300', 'avg_voltage_300',
                'avg_current_600', 'avg_voltage_600',
                'avg_current_all', 'avg_voltage_all','BmsSoc']].iloc[:-time_pred]
        X_curr = X['Curr'].values
        X_volt = X['Volt'].values
        X_curr = filter(X_curr.reshape(-1, 1), [70, 30, 10]).reshape(-1, )
        X_volt = filter(X_volt.reshape(-1, 1), [30, 20, 5]).reshape(-1, )
        # # 可视化
        # series_list = [
        #     [X_curr, X['Curr'].values],
        #     [X_volt, X['Volt'].values],
        # ]
        # series_name_list = [
        #     # ['pred_refrigerant_mix_temp', 'refrigerant_mix_temp'],
        #     ['过滤后电流', '过滤前电流'],
        #     ['过滤后电压', '过滤前电压'],
        # ]
        # pic_name = os.path.basename(file_path).replace('.csv', '.png')
        # draw_pred(series_list, series_name_list, '../data/data_anyls/curr_voltage_filter', f'{pic_name}')

        X['Curr'] = X_curr
        X['Volt'] = X_volt

        y = df['TrueSoc'].values[time_pred:]
        L_length = len(X) - time_step - time_pred
        if L_length < 500:
            continue

        for i in range(0,L_length,time_skip):
            select_every_slip = range(i, i + time_step, time_slip)
            # 确保 select_every_slip 的范围不会超过序列的长度
            if max(select_every_slip) < L_length:
                x_all = X.iloc[select_every_slip]
                Volt = x_all['Volt'].values
                Curr = x_all['Curr'].values
                Ah = x_all['Ah'].values
                P_vxi = Volt * Curr
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
                x_in =  ([Volt] + [Curr] + [Ah] + [P_vxi] +
                         [avg_current_60] + [avg_voltage_60] +
                         [avg_current_300] + [avg_voltage_300] +
                         [avg_current_600] + [avg_voltage_600] +
                         [avg_current_all] + [avg_voltage_all] + [BmsSoc])
                x_in = np.array(x_in).T
                y_out = np.array(y[i + time_step])
                for i in range(1,time_pred):
                    y_out = np.append(y_out, y[i + time_step + 1])

                X_split.append(x_in)
                y_split.append(y_out)


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
    def __init__(self, all_data_path, time_step=300, time_slip=1, time_skip=5, return_point=True):
        self.all_data_path = all_data_path
        self.time_step = time_step
        self.time_slip = time_slip
        self.return_point = return_point
        self.time_skip = time_skip

        self.all_x = []
        self.all_y = []
        self.info = []

        self.all_x , self.all_y, self.info = merge_data(self.all_data_path, self.time_step, self.time_slip, self.time_skip, self.return_point)

    def __len__(self):
        return len(self.all_x)

    def __getitem__(self, item_index):
        x = self.all_x[item_index].astype('float32')
        y = self.all_y[item_index].astype('float32')
        return x, y