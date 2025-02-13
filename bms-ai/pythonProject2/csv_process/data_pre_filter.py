# 导入需要用的包
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle
import torch
from PycharmProjects.pythonProject2.dataset.dataset_soc import SOCBaseDataset

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



def calculate_moving_average(series, window_size):
    """计算移动平均值"""
    return series.rolling(window=window_size, min_periods=1).mean()

def data_prepare1(T):
    L = len(T['Curr'])
    if L < 300 :
        return pd.DataFrame([])
    T['avg_current_all'] = calculate_moving_average(T['Curr'], window_size=L)
    T['avg_voltage_all'] = calculate_moving_average(T['Volt'], window_size=L)
    T['avg_current_300'] = calculate_moving_average(T['Curr'], window_size=300)
    T['avg_voltage_300'] = calculate_moving_average(T['Volt'], window_size=300)
    T['avg_current_600'] = calculate_moving_average(T['Curr'], window_size=600)
    T['avg_voltage_600'] = calculate_moving_average(T['Volt'], window_size=600)



    condition1 = (T['BmsSoc'] >= 70.0)
    if condition1.any():
        idx_begin1 = np.where(condition1)[0][0]
    else:
        idx_begin1 = 0

    idx_begin = idx_begin1

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


# 主函数
def main():
    # 需要修改的地方1/7 输入原始csv的路径
    input_folder_path = 'D:/Code/code/submit/bms_ai/PycharmProjects/data/filter_data/data_handle_new/'  # 替换为你的输入文件夹路径
    # 需要修改的地方2/7 输出过滤csv的路径
    output_folder_path = 'D:/Code/code/submit/bms_ai/PycharmProjects/data/filter_data/data_handle_BmsSoc_10_filter_v1'  # 替换为你想要保存处理后文件的文件夹路径
    # 需要修改的地方3/7 储存csv每个文件的title前部分
    name_title = 'data_handle_BmsSoc_10_filter_v1'
    # 需要修改的地方4/7 存储pkl的路径
    pkl_folder_path = "D:/Code/code/submit/bms_ai/PycharmProjects/data/filter_data/pkl_data"
    # 需要修改的地方5/7 训练集pkl的文件名
    train_dataset_path = os.path.join(pkl_folder_path, "seed111_1127_train_dataset_1000.pkl")
    # 需要修改的地方6/7 测试集pkl的文件名
    eval_dataset_path = os.path.join(pkl_folder_path, "seed111_1127_eval_dataset_1000.pkl")
    # 需要修改的地方7/7 选择多少个pkl，choose_len == -1 代表全部
    choose_len =1000

 # 替换目标pkl 存储的位置
    # 需要确认的地方1
    time_step = 300 # 输入时间跨度
    # 需要确认的地方2
    time_slip = 5 # 输入时间间隔

    #输入时间维度 = time_step/time_slip
    # 需要确定地方3 是否要生成过滤的csv,如果已经生成过了，就不用再次生成
    Flag_csv_filter = True

    # 1先将数据进行清洗 datahandl1 + datahandl2

    if Flag_csv_filter:
        # 创建输出文件夹（如果不存在）
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        i = 0
        all_inputdata_path = os.listdir(input_folder_path)
        idx = 0
        for filename in tqdm(all_inputdata_path):
            if idx >= 20000:
                break
            if filename.endswith('.csv') and (i >-1) and i < (len(all_inputdata_path)):
                i = i + 1
                file_path = os.path.join(input_folder_path, filename)

                # 读取CSV文件
                df = pd.read_csv(file_path)
                # 处理数据
                processed_df = data_prepare1(df)
                if (len(processed_df) >= time_step):

                    output_filename = f"{name_title}_{filename}"
                    output_file_path = os.path.join(output_folder_path, output_filename)
                    processed_df.to_csv(output_file_path, index=False)
                    print(f"Processed and saved {filename} to {output_filename}")
                    idx += 1
                else:
                    print(filename)

    # 2生成训练集和验证集的pkl

    if not os.path.exists(pkl_folder_path):
        os.makedirs(pkl_folder_path)


    all_outputdata_name = os.listdir(output_folder_path)
    all_outputdata_path = [os.path.join(output_folder_path, file_name) for file_name in all_outputdata_name]
    seed = 111
    train_eval_rate = 0.5
    np.random.seed(seed)
    all_data_index = np.arange(len(all_outputdata_path))
    np.random.shuffle(all_data_index)
    torch.manual_seed(seed)

    train_final_place = int(train_eval_rate * len(all_data_index))

    if choose_len == -1:
        train_index = all_data_index[:train_final_place]
        eval_index = all_data_index[train_final_place:]
    else:
        train_index = all_data_index[train_final_place-choose_len:train_final_place]
        eval_index = all_data_index[train_final_place:train_final_place+choose_len]

    train_data_path = [all_outputdata_path[i] for i in train_index]
    eval_data_path = [all_outputdata_path[i] for i in eval_index]

    train_dataset = SOCBaseDataset(train_data_path,time_step=time_step,time_slip=time_slip,return_point=True)
    with open(train_dataset_path, 'wb') as f:
        pickle.dump(train_dataset, f)
        print('trian_dataset_len= ',len(train_dataset.all_x))

    eval_dataset = SOCBaseDataset(eval_data_path,time_step=time_step,time_slip=time_slip,return_point=False)
    with open(eval_dataset_path, 'wb') as f:
        pickle.dump(eval_dataset, f)
        print('trian_dataset_len= ', len(eval_dataset.all_x))
        print("pkl_data_save_success")
    # else:
    #     with open(train_dataset_path, 'rb') as f:
    #         train_dataset = pickle.load(f)
    #
    #     with open(eval_dataset_path, 'rb') as f:
    #         eval_dataset = pickle.load(f)


if  __name__ == '__main__':
    main()