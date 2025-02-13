from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import os


def merge_data(file_path_list, time_step, time_slip=5, return_point=False):
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
        #组合填充
        df = ffill_df.combine_first(bfill_df)

        avg_current_60 = df['Curr'].rolling(window=60, min_periods=1).mean()
        avg_voltage_60 = df['Volt'].rolling(window=60, min_periods=1).mean()
        X = df[['Volt', 'Temp', 'Curr','Ah',
                'avg_current_all', 'avg_voltage_all','BmsSoc']].copy()
        X.loc[:, 'avg_current_60'] = avg_current_60
        X.loc[:, 'avg_voltage_60'] = avg_voltage_60
        y = df['TrueSoc']

        # for i in range(len(X) - time_step):
        #     if X['Curr'][i] - X['Curr'][i-1]

        for i in range(len(X) - time_step):
            select_every_slip = range(i, i + time_step, time_slip)
            # 确保 select_every_slip 的范围不会超过 X 的长度
            if max(select_every_slip) < len(X):
                x_all = X.iloc[select_every_slip]
                Volt = x_all['Volt'].values
                Temp = x_all['Temp'].values
                Curr = x_all['Curr'].values
                Ah = x_all['Ah'].values
                P_vxi = Volt * Curr
                avg_current_60 = x_all['avg_current_60'].values
                avg_voltage_60 = x_all['avg_voltage_60'].values


                BmsSoc = x_all['BmsSoc'].values

                # 将特征列表转换成一个数组
                x_in =  ([Volt] + [Curr] + [Ah] + [P_vxi]+
                         [avg_current_60] + [avg_voltage_60] + [BmsSoc])
                x_in = np.array(x_in).T

                if np.isnan(x_in).any():
                    print('nan')
                    print(x_in)
                X_split.append(x_in)
                y_split.append(y[i + time_step])

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
    def __init__(self, all_data_path, time_step=300, time_slip=5, return_point=True):
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
