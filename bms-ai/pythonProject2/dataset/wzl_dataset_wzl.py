import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from scipy.interpolate import interp2d, interp1d
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import median_filter

def filter(x, sizes):
    filtered_signal = x.copy()
    for size in sizes:
        filtered_signal = median_filter(filtered_signal, size=size)
    return filtered_signal
    
def z_score(data, mean, std):
    normed_data = (data - mean) / (std)
    return normed_data

def add_noise(x, mean=0, std=0.01):
    noise = np.random.normal(mean, std, x.shape)
    return x + noise

def uniform_sampling(data, shape_1, num_points=60):
    original_length = data.shape[0]
    if original_length == num_points:
        return data
    
    # 生成原始数据的索引
    original_indices = np.arange(original_length)
    
    # 生成新的均匀分布的索引
    new_indices = np.linspace(0, original_length - 1, num_points)
    
    # 对第一个维度进行线性插值
    interpolated_data = np.zeros((num_points, shape_1))
    for i in range(shape_1):
        interpolator = interp1d(original_indices, data[:, i], kind='linear')
        interpolated_data[:, i] = interpolator(new_indices)
    
    return interpolated_data

def read_csv(data_path):
    data = pd.read_csv(data_path)

    gt_soc = np.array(data['TrueSoc'])
    volt = np.array(data['Volt'])
    tmp = np.array(data['Temp'])
    curr = np.array(data['Curr'])
    bms_soc = np.array(data['BmsSoc'])
    avg_voltage = np.array(data["avg_voltage"])
    avg_current = np.array(data["avg_current"])
    ah = np.array(data["Ah"])

    out_dict = {
        "gt_soc" : gt_soc,
        "volt" : volt,
        "tmp" : tmp,
        "curr" : curr,
        "bms_soc" : bms_soc,
        "avg_voltage" : avg_voltage,
        "avg_current" : avg_current,
        "ah" : ah
        }
    return out_dict

class train_dataset_soc(torch.utils.data.Dataset):

    def __init__(self, all_data_path):
        self.all_data_path = all_data_path

        self.all_x = []
        self.all_y = []
        count = 0
        for path_index in tqdm(range(len(self.all_data_path))):
            count += 1
            if count >= 5000:
                break
            data_path = self.all_data_path[path_index]
            out_dict = read_csv(data_path)
            # ---------------input--------------------------
            gt_soc = out_dict["gt_soc"]                   
            volt = out_dict["volt"]
            tmp = out_dict["tmp"]
            curr = out_dict["curr"]
            avg_v = out_dict["avg_voltage"]
            avg_curr = out_dict["avg_current"]
            p_iu = volt * curr            
            ah = out_dict["ah"]
            bms_soc = out_dict["bms_soc"]
            #------------------------数据处理---------------------
            curr = filter(curr.reshape(-1, 1), [70, 30, 10]).reshape(-1,)
            volt = filter(volt.reshape(-1, 1), [30, 20, 5]).reshape(-1,)

            # ah = np.gradient(ah.flatten(), 1)
            # ---------------output-------------------------
            x = [volt] + [tmp] + [curr] + [p_iu] + [avg_v] + [avg_curr] + [ah]
            y = [gt_soc] + [bms_soc]

            x = np.array(x).T
            y = np.array(y).T

            x = x[:-300]
            y = y[300:]
            #----------------------------------------
            mean_list = [3463.985985, 38.282940, 81.054740, 281342.280081, 3469.481652, 97.281070, 124.407585]
            std_list = [22.570149, 2.487525, 32.090257, 112728.841726, 19.137243, 32.608585, 22.388481]
            for i in range(x.shape[0]):
                for j in range(7):
                    x[i][j] = z_score(x[i][j], mean_list[j], std_list[j])
            #-------------------取时序窗口---------------------------------
            _x = []
            _y = []
            win_len = 300
            step = 10
            y_len = 2
            for i in range(0, len(x)-win_len, step):
                _x.append(x[i:i+win_len])
                # _x.append(uniform_sampling(x[i:i+win_len], 7, 60))
                _y.append(y[i:i+y_len])
            x = np.array(_x).reshape(-1, 300, 7)
            y = np.array(_y).reshape(-1, y_len, 2)
            #---------------------------------------------------------------
            self.all_x.append(x)
            self.all_y.append(y)
        self.all_x = np.concatenate(self.all_x)        
        self.all_y = np.concatenate(self.all_y)
        # self.all_x = add_noise(self.all_x)
        
        # test_x = self.all_x.reshape(-1, 7)
        # for i in range(7):
        #     print(f"x{i}_"*10)
        #     print(np.mean(test_x[:, i]), np.std(test_x[:, i]))
        # exit()

    def __len__(self):
        return len(self.all_x)

    def __getitem__(self, item_index):        

        x = self.all_x[item_index]
        y = self.all_y[item_index]
        return x, y
        
class eval_dataset_soc(torch.utils.data.Dataset):
    def __init__(self, all_data_path):
        self.all_data_path = all_data_path
        self.all_x = []
        self.all_y = []
        self.all_info = []
        count = 0
        for path_index in tqdm(range(len(self.all_data_path))):
            count += 1
            if count >= 100:
                break
            data_path = self.all_data_path[path_index]
            data = read_csv(data_path)
            out_dict = pd.DataFrame(data)

            # ---------------input--------------------------
            out_dict["avg_curr"] = out_dict["curr"].rolling(window=300, min_periods=1).mean()
            out_dict["avg_v"] = out_dict["volt"].rolling(window=300, min_periods=1).mean()

            condition1 = out_dict['gt_soc'] >= 80.0
            if condition1.any():
                idx_begin1 = np.where(condition1)[0][0]
            else:
                idx_begin1 = 0


            out_dict = out_dict_old.iloc[idx_begin1-300:, :]

            gt_soc = out_dict["gt_soc"].values
            volt = out_dict["volt"]
            tmp = out_dict["tmp"].values
            curr = out_dict["curr"]
            # avg_v = out_dict["avg_voltage"]
            # avg_curr = out_dict["avg_current"]
            avg_curr = curr.rolling(window=300, min_periods=1).mean()
            avg_v = volt.rolling(window=300, min_periods=1).mean()

            volt = volt.values
            curr = curr.values
            p_iu = volt * curr            
            ah = out_dict["ah"].values
            bms_soc = out_dict["bms_soc"].values



            #------------------------数据处理---------------------
            curr = filter(curr.reshape(-1, 1), [70, 30, 10]).reshape(-1,)
            volt = filter(volt.reshape(-1, 1), [30, 20, 5]).reshape(-1,)

            # ah = np.gradient(ah.flatten(), 1)
            # ---------------output-------------------------
            x = [volt] + [tmp] + [curr] + [p_iu] + [avg_v] + [avg_curr] + [ah]
            y = [gt_soc] + [bms_soc]

            x = np.array(x).T
            y = np.array(y).T

            x = x[:-300]
            y = y[300:]
            #----------------------------------------
            mean_list = [3463.985985, 38.282940, 81.054740, 281342.280081, 3469.481652, 97.281070, 124.407585]
            std_list = [22.570149, 2.487525, 32.090257, 112728.841726, 19.137243, 32.608585, 22.388481]
            for i in range(x.shape[0]):
                for j in range(7):
                    x[i][j] = z_score(x[i][j], mean_list[j], std_list[j])
            #-------------------取时序窗口---------------------------------
            # _x = []
            # _y = []
            # win_len = 60
            # for i in range(0, len(x)-win_len):
            #     all_len += 1
            #     _x.append(x[i:i+win_len])
            #     _y.append(y[i])
            # if all_len >= 10000:
            #     break
            # x = np.array(_x).reshape(-1, 60, 7)
            # y = np.array(_y).reshape(-1, 1)
            #---------------------------------------------------------------
            _x = []
            _y = []
            win_len = 300
            step = 2
            y_len = 2
            for i in range(0, len(x)-win_len, step):
                _x.append(x[i:i+win_len])
                # _x.append(uniform_sampling(x[i:i+win_len], 7, 60))
                _y.append(y[i:i+y_len])
            x = np.array(_x).reshape(-1, 300, 7)
            try:
                y = np.array(_y).reshape(-1, y_len, 2)
            except:
                print('error')

            #---------------------------------------------------------------
            self.all_x.append(x)
            self.all_y.append(y)
            self.all_info.append(data_path)

    def __len__(self):
        return len(self.all_x)

    def __getitem__(self, item_index):        

        x = self.all_x[item_index]
        y = self.all_y[item_index]
        return x, y