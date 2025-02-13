import os
from tqdm import tqdm
import numpy as np
import torch

from AC_energy_pred import config_all
from AC_energy_pred.data_utils import read_csv


# 最大最小归一化
def norm_data(data, max_data, min_data):
    normed_data = (data - min_data) / (max_data - min_data)
    return normed_data


def recover_data(normed_data, max_data, min_data):
    data = normed_data * (max_data - min_data) + min_data
    return data


# 根据开度估算制冷剂流量 # todo 未完成待定
def cal_refrigerant_vol_para(battery_cooling_status_act_pos, cab_cooling_status_act_pos, cab_heating_status_act_pos, hp_mode, compressor_speed):
    refrigerant_vol_para = np.zeros_like(hp_mode)
    refrigerant_vol_para[hp_mode == config_all.heat_mode_index] = cab_heating_status_act_pos[hp_mode == config_all.heat_mode_index]
    refrigerant_vol_para[hp_mode == config_all.cooling_mode_index] = cab_cooling_status_act_pos[hp_mode == config_all.cooling_mode_index]
    return refrigerant_vol_para


# 制热模式数据 使用数据规则得到的ags开度风扇占空比 作为学习目标
class AcPidOutBaseDataset(torch.utils.data.Dataset):

    def __init__(self, all_data_path, return_point):
        self.all_data_path = all_data_path
        self.return_point = return_point

        self.all_x = []
        self.all_y = []
        self.all_info = []

        for path_index in tqdm(range(len(self.all_data_path))):
            data_path = self.all_data_path[path_index]
            out_dict = read_csv(data_path)

            # 上一时刻ac_pid_pout
            # AC_PID_out
            lst_ac_pid_out_hp = out_dict['ac_pid_out_hp'][:-1]

            # 当前环境温度
            # TmRteOut_AmbTEstimd   temp_amb
            temp_amb = out_dict['temp_amb'][1:]
            # 当前主驾驶设定温度
            # HvacFirstLeTSt HMI_Temp_FL
            cab_fl_set_temp = out_dict['cab_fl_set_temp'][1:]
            # 当前副驾驶设定温度
            # HvacFirstRiTSt HMI_Temp_FR
            cab_fr_set_temp = out_dict['cab_fr_set_temp'][1:]
            # 当前后排设定温度
            # cab_rl_set_temp = out_dict['cab_rl_set_temp'][1:]

            # 当前主驾驶目标进风温度
            # SEN_Duct_FL
            # cab_fl_req_temp = out_dict['cab_fl_req_temp'][1:]
            # # 当前副驾驶目标进风温度
            # # SEN_Duct_FR
            # cab_fr_req_temp = out_dict['cab_fr_req_temp'][1:]
            # # 当前后排目标进风温度
            # # SEN_Duct_FR
            # cab_rl_req_temp = out_dict['cab_rl_req_temp'][1:]

            # 上一时刻饱和低压
            # LoSideP
            lo_pressure = out_dict['lo_pressure'][:-1]
            # 上一时刻饱和高压
            # HiSideP
            hi_pressure = out_dict['hi_pressure'][:-1]
            # 上一时刻目标饱和低压
            # RefrigLoPTar
            aim_lo_pressure = out_dict['aim_lo_pressure'][:-1]
            # 上一时刻目标饱和高压
            # RefrigHiPTar
            aim_hi_pressure = out_dict['aim_hi_pressure'][:-1]

            # 当前乘务舱温度
            temp_in_car = out_dict['temp_in_car'][1:]
            #电池请求冷却液温度
            temp_battery_req = out_dict['temp_battery_req'][1:]
            #冷却液进入电池的温度
            temp_coolant_battery_in = out_dict['temp_coolant_battery_in'][1:]

            # ac_kp_rate
            ac_kp_rate_last = out_dict['ac_kp_rate'][:-1]

            # 压缩机排气温度
            temp_p_h_2 = out_dict['temp_p_h_2'][:-1]
            # 内冷温度
            temp_p_h_5 = out_dict['temp_p_h_5'][:-1]
            # 压缩机进气温度
            temp_p_h_1_cab_heating = out_dict['temp_p_h_1_cab_heating'][:-1]
            # 蒸发器温度
            temp_evap = out_dict['temp_evap'][:-1]

            # 空调模式
            hp_mode_ac = out_dict['hp_mode_ac'][:-1]
            # 制冷模式下膨胀阀开度
            cab_cooling_status_act_pos = out_dict['cab_cooling_status_act_pos'][:-1]

            #     ac_pid_pout_hp = np.array(data['AC_PID_Pout_HP'])   # 压缩机转速比例输出  未找到
            #     ac_pid_iout_hp = np.array(data['AC_PID_Iout_HP'])   # 压缩机转速积分输出
            #     ac_pid_dout_hp = np.array(data['AC_PID_Dout_HP'])   # 压缩机转速微分输出
            # Ac_pid_pout
            # ac_pid_pout_hp = out_dict['ac_pid_pout_hp'][:-1]
            # # Ac_pid_iout
            # ac_pid_iout_hp = out_dict['ac_pid_iout_hp'][:-1]
            # # Ac_pid_dout
            # ac_pid_dout_hp = out_dict['ac_pid_dout_hp'][:-1]


            # 输出
            # 下一时刻acc_pid_out_hp
            ac_pid_out_hp = out_dict['ac_pid_out_hp'][1:]

            # ac_kp_rate
            ac_kp_rate = out_dict['ac_kp_rate'][1:]


            # 过滤条件
            # 只考虑模式10
            hp_mode = out_dict['hp_mode'][1:]
            # 不考虑过热度为负的情况
            sh_act_mode_10 = out_dict['sh_act_mode_10'][1:]

            # 上一时刻压缩机转速
            compressor_speed_last = out_dict['compressor_speed'][:-1]

            #当前压缩机转速
            compressor_speed = out_dict['compressor_speed'][1:]
            # 只考虑压缩机速率变化在1150r/s，且高于正常工作的最低转速800的情况
            compressor_speed_change = abs(compressor_speed - compressor_speed_last)

            # 上一时刻ac_pid_pout 当前环境温度
            # 当前主驾设定温度 当前副驾设定温度 当前后排设定温度
            # 当前主驾目标进风温度 当前副驾目标进风温度 当前后排目标进风温度
            # 上一时刻饱和低压 上一时刻饱和高压 上一时刻目标饱和低压 上一时刻目标饱和高压
            # x = ([lst_ac_pid_out_hp] +
            #      [lo_pressure] + [hi_pressure] + [aim_hi_pressure] +
            #      [temp_p_h_2] + [temp_p_h_1_cab_heating] + [temp_p_h_5])
            x = ([lst_ac_pid_out_hp] +
                 [temp_amb] + [cab_fl_set_temp] + [cab_fr_set_temp] +
                 [lo_pressure] + [hi_pressure] + [aim_lo_pressure] + [aim_hi_pressure] +
                 [temp_in_car] + [temp_battery_req] + [temp_coolant_battery_in] +
                 [temp_p_h_2] + [temp_evap] + [hp_mode_ac] + [cab_cooling_status_act_pos])
                 # [ac_pid_pout_hp] + [ac_pid_iout_hp] + [ac_pid_dout_hp])

            # x = ([lst_ac_pid_out_hp] +
            #      [temp_amb] + [cab_fl_set_temp] + [cab_fr_set_temp] +
            #      [lo_pressure] + [hi_pressure] + [aim_lo_pressure] + [aim_hi_pressure] +
            #      [temp_incar] + [temp_battery_req] + [temp_coolant_battery_in] + [ac_kp_rate_last] +
            #      [temp_p_h_2] + [temp_p_h_5] + [temp_p_h_1_cab_heating] +  [temp_evap] + [hp_mode_ac] + [cab_cooling_status_act_pos])


            x = np.array(x).T
            # mask = (wind_vol <= 50) | ((hp_mode != config_all.heat_mode_index)) | (
            #         warmer_p > 0) | (cab_cooling_status_act_pos != 0)
            # mask = (hp_mode != config_all.heat_mode_index) | (compressor_speed_last < 800)| (compressor_speed_change > 1150)
            # mask = (compressor_speed_last < 0)
            mask = (hp_mode != config_all.heat_mode_index) | (sh_act_mode_10 < 0) | (compressor_speed_change > 1150) | (
                        compressor_speed_last < 800)
            # | ((aim_hi_pressure - hi_pressure)*0.0244 >0)

            # mask = (hp_mode != config_all.heat_mode_index)
            print(f"x shape:{x.shape}")
            print(f"mask shape:{mask.shape}")
            mask_x = np.ma.array(x, mask=np.repeat(mask.reshape(-1, 1), x.shape[-1], axis=-1))
            # mask_y = ma.array(y, mask=np.repeat(mask.reshape(-1,1),y.shape[-1],axis=-1))
            clumps = np.ma.clump_unmasked(mask_x[:, 0])



            y = [ac_pid_out_hp]
            # y = [ac_kp_rate]

            y = np.array(y).T

            all_split_x = []
            all_split_y = []
            for split_index in range(len(clumps)):
                split_range = clumps[split_index]

                split_x = x[split_range]
                split_y = y[split_range]

                if len(split_x) < config_all.min_split_len:
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
            # self.all_info = np.concatenate(self.all_info)

    def __len__(self):
        return len(self.all_x)

    def __getitem__(self, item_index):

        x = self.all_x[item_index]
        y = self.all_y[item_index]

        return x, y
