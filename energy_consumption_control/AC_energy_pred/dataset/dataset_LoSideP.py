import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import config_all as config
from data_utils import read_csv

class dataset_LoSideP(torch.utils.data.Dataset):

    def __init__(self, all_data_path, return_point):
        self.all_data_path = all_data_path
        self.return_point = return_point

        self.all_x = []
        self.all_y = []
        self.all_info = []


        for path_index in tqdm(range(len(self.all_data_path))):

            data_path = self.all_data_path[path_index]
            out_dict = read_csv(data_path)

            # ---------------input--------------------------
            TmRteOut_InrCondOutlT = out_dict["temp_p_h_5"]                     #内冷温度
            TmRteOut_HiSideP = out_dict["hi_pressure"]                         #饱和高压
            sTmSigIn_X_CexvActPosn = out_dict["cab_heating_status_act_pos"]    #CEXV开度 制热
            Ags_Openness = out_dict["ags_openness"]                            #AGS开度
            CFan_PWM = out_dict["cfan_pwm"]                                    #冷却风扇转速
            COMM_VehicleSpd = out_dict["car_speed"]                            #车速
            TmRteOut_AmbTEstimd = out_dict["temp_amb"]                         #环境温度
            # ---------------output-------------------------
            TmRteOut_LoSideP = out_dict["lo_pressure"]                         #饱和低压
            lo_pressure_temp = out_dict["temp_p_h_1_cab_heating"]              #焓1温度
            
            wind_vol = out_dict['wind_vol']
            hp_mode = out_dict['hp_mode']
            hvac_mode = out_dict['hvac_mode']
            cab_cooling_status_act_pos = out_dict['cab_cooling_status_act_pos']
            warmer_p = out_dict['warmer_p']

            x = [TmRteOut_InrCondOutlT] + [TmRteOut_HiSideP] + [sTmSigIn_X_CexvActPosn] + \
                [Ags_Openness] + [CFan_PWM] + [COMM_VehicleSpd] + [TmRteOut_AmbTEstimd]
            x = np.array(x).T

            y = [TmRteOut_LoSideP] + [lo_pressure_temp]
            y = np.array(y).T

        
            # 排除风量为0或电流为0的数据 空调模式不是2 乘员舱模式为3
            # for ind in range(len(out_dict['wind_vol'])):
            #     if (wind_vol[ind] > 50) and (cab_cooling_status_act_pos[ind] == 0) and (hp_mode[ind] == config.heat_mode_index) and (warmer_p[ind] <= 0):
            #         self.x.append([TmRteOut_InrCondOutlT[ind],TmRteOut_HiSideP[ind],sTmSigIn_X_CexvActPosn[ind],Ags_Openness[ind],CFan_PWM[ind],COMM_VehicleSpd[ind], TmRteOut_AmbTEstimd[ind]])
            #         self.y.append([TmRteOut_LoSideP[ind],lo_pressure_temp[ind]])
            # self.all_x.extend(self.x)
            # self.all_y.extend(self.y)
            #---------------------------------------------------------------------------------
            # mask = (wind_vol <= 50) | ((hp_mode != config.heat_mode_index)) | (
            #             warmer_p > 0) | (cab_cooling_status_act_pos != 0)
            #筛选数据
            mask = (wind_vol <= 50) | (hp_mode != 10)
            mask_x = np.ma.array(x, mask=np.repeat(mask.reshape(-1, 1), x.shape[-1], axis=-1))
            clumps = np.ma.clump_unmasked(mask_x[:, 0])

            all_split_x = []
            all_split_y = []
            for split_index in range(len(clumps)):
                split_range = clumps[split_index]

                split_x = x[split_range]
                split_y = y[split_range]

                if len(split_x) < config.min_split_len:
                    continue

                all_split_x.append(split_x.astype('float32'))
                all_split_y.append(split_y.astype('float32'))

            if True in np.isnan(x) or True in np.isnan(y):
                print(path_index, data_path, 'error')
                continue

            if len(all_split_x) == 0:
                print(path_index, data_path, 'no ok split')
                continue

            if return_point:
                self.all_y.append(np.concatenate(all_split_y))
                self.all_x.append(np.concatenate(all_split_x))
            else:
                self.all_y.append(all_split_y)
                self.all_x.append(all_split_x)
                self.all_info.append([f'{data_path}_{i}' for i in range(len(clumps))])

        if return_point:
            self.all_x = np.concatenate(self.all_x)
            self.all_y = np.concatenate(self.all_y)
            
    def __len__(self):
        return len(self.all_x)

    def __getitem__(self, item_index):        

        x = self.all_x[item_index]
        y = self.all_y[item_index]
        return x, y