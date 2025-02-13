# 导入需要用的包
import pandas as pd
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

# ### 真实SOC计算
def pro_charging_trueSocCalc(charging_data, time, SOC_end, Q_ah):
    L = len(charging_data['hvbattactcur'])
    ah_soc = np.zeros(L)
    ah=np.zeros(L)
    for k in range(1,L):
        dt = time[k] - time[k-1]
        ah_soc[k] = ah_soc[k-1] + charging_data['hvbattactcur'][k]*dt/Q_ah/36
        ah[k]=ah[k-1]-charging_data['hvbattactcur'][k]*dt/3600
    TrueSoc = -ah_soc + SOC_end + ah_soc[-1]
    return TrueSoc, ah
'''
def data_prepare(T):
    startSOC = 70  # 起点SOC,
    endSOC = 99  # 终点SOC,
    endVolt = 3.75  # 终点电压, V
    Q_ah = 188.8
    CurrMean = -50
    T_each_all = pd.DataFrame()
    if(len(T)<100):
        return T_each_all
    for p in range(len(T) - 1, 0, -1):
        tmp_dt = T['businesstimestamp'][p] - T['businesstimestamp'][p - 1]
        tmp1 = (tmp_dt > -1) & (tmp_dt < 3100)

        tmp = T['hvbattcellmaxsoc'][p]
        tmp2 = (tmp >= 0) & (tmp <= 100)

        tmp = T['hvbattcelltmin'][p]
        tmp3 = (tmp > -40) & (tmp < 60)
        tmp = T['hvbattcelltmax'][p]
        tmp3 = (tmp3) & (tmp > -40) & (tmp < 60)

        tmp = T['hvbattucellmax'][p]
        tmp4 = (tmp > 2) & (tmp < 4.5)

        tmp = T['hvbattactcur'][p]
        tmp5 = (tmp > -800) & (tmp < 1000)

        tmp = tmp1 & tmp2 & tmp3 & tmp4 & tmp5
        if not tmp:
            #print(T['vid'][0])
            break

    T = T.iloc[p:-1, :].reset_index(drop=True)
    # flg1 ,flg2, flg3, flg4, flg5 = False
    if len(T) > 100:
        flg1 = T['hvbattcellmaxsoc'][0] < startSOC  # 起点SOC
        flg2 = T['hvbattcellmaxsoc'].tail(1).values >= endSOC  # 终点SOC
        flg3 = T['hvbattucellmax'].tail(1).values > endVolt  # 终点电压
        flg4 = T['hvbattactcur'][0] > 0  # 起点电流
        flg5 = T['hvbattactcur'].mean() < CurrMean
    else:
        print('step 1 invalid')
        return T_each_all

    if flg1 & flg2 & flg3 & flg4 & flg5 :
        time = T['businesstimestamp']/1000 - T['businesstimestamp'][0]/1000
        CurrRate = T['hvbattactcur'] / -Q_ah
        #TrueSoc = pro_charging_trueSocCalc(T, time,T['hvbattcellmaxsoc'].tail(1).values, Q_ah)
        TrueSoc, ah = pro_charging_trueSocCalc(T, time,100, Q_ah)
        new_data = pd.DataFrame({'time': time, 'CurrRate': CurrRate, 'TrueSoc': TrueSoc,'Ah':ah})
        T = pd.concat([T, new_data], axis=1)

        # 开始充电的起点
        idx_begin = 0
        for i in range(len(T['hvbattactcur'])):
            if T['hvbattactcur'][i] < -1:
                idx_begin = i
                break

        T = T.iloc[idx_begin:-1, :].reset_index(drop=True)
        T['time'] = T['time'] - T['time'][0]
        T_each_all = pd.DataFrame({
            'time': T['time'],
            'TrueSoc': T['TrueSoc'],
            'Volt': T['hvbattucellmax'] *1000,
            'Temp': (T['hvbattcelltmin'] + T['hvbattcelltmax']) / 2,
            'Curr': T['hvbattactcur'] * -1,
            'BmsSoc': T['hvbattcellmaxsoc'],
            'Ah':T['Ah']
           # 'vid': T['vid'],
           # 'flgIni': np.where(T.index == 0, True, False).astype(bool)

        })
        return T_each_all
    else:
        print('step 2 invalid')
        return T_each_all
'''

def data_prepare(T):
    startSOC = 70  # 起点SOC,
    endSOC = 99  # 终点SOC,
    endVolt = 3.75  # 终点电压, V
    Q_ah = 188.8
    CurrMean = -50
    T_each_all = pd.DataFrame({'time': [],'TrueSoc': [],'Volt': [],'Temp': [],'Curr': [],'BmsSoc': [],'flgIni': []})
    # 时间戳异常排查
    tmp_dt = T['businesstimestamp'].diff()[1:]
    tmp1 = (tmp_dt > -1).all() & (tmp_dt < 3100).all()
    # SOC异常排查
    tmp2 = (T['hvbattcellmaxsoc'] >= 0).all() & (T['hvbattcellmaxsoc'] <= 100).all()
    # 温度异常排查
    tmp3 = (T['hvbattcelltmin'] > -40).all() & (T['hvbattcelltmin'] < 60).all() & (T['hvbattcelltmax'] > -40).all() & (
                T['hvbattcelltmax'] < 60).all()
    # 电压异常排查
    tmp4 = (T['hvbattucellmax'] > 2).all() & (T['hvbattucellmax'] < 4.5).all()
    # 电流异常排查
    tmp5 = (T['hvbattactcur'] > -800).all() & (T['hvbattactcur'] < 1000).all()

    tmp = tmp1 & tmp2 & tmp3 & tmp4 & tmp5
    if tmp:
        flg1 ,flg2, flg3, flg4, flg5 = [False] * 5
        if len(T) > 100:
            flg1 = T['hvbattcellmaxsoc'][0] < startSOC  # 起点SOC
            flg2 = T['hvbattcellmaxsoc'].tail(1).values >= endSOC  # 终点SOC
            flg3 = T['hvbattucellmax'].tail(1).values > endVolt  # 终点电压
            flg4 = T['hvbattactcur'][0] > 0  # 起点电流
            flg5 = T['hvbattactcur'].mean() < CurrMean

        if flg1 & flg2 & flg3 & flg4 & flg5 :
            time = T['businesstimestamp']/1000 - T['businesstimestamp'][0]/1000
            CurrRate = T['hvbattactcur'] / -Q_ah
            TrueSoc, Ah = pro_charging_trueSocCalc(T, time,100, Q_ah)
            new_data = pd.DataFrame({'time': time, 'CurrRate': CurrRate, 'TrueSoc': TrueSoc,'Ah': Ah})
            T = pd.concat([T, new_data], axis=1)

            # 开始充电的起点
            idx_begin = 0
            for i in range(len(T['hvbattactcur'])):
                if T['hvbattactcur'][i] < -1:
                    idx_begin = i
                    break
            T = T.iloc[idx_begin:-1, :].reset_index(drop=True)
            T_each_all = pd.DataFrame({
                'time': T['time'],
                'TrueSoc': T['TrueSoc'],
                'Volt': T['hvbattucellmax'],
                'Temp': (T['hvbattcelltmin'] + T['hvbattcelltmax']) / 2,
                'Curr': T['hvbattactcur'] * -1,
                'BmsSoc': T['hvbattcellmaxsoc'],
                'Ah': T['Ah']
                # 'ReqCurr' : T['hvbattsopmdllongtichrgcurr']
            })
    return T_each_all
# 主函数
def main():
    input_folder_path = 'D:/Code/code/submit/bms_ai/PycharmProjects/data/badcase_csv/'  # 替换为你的输入文件夹路径
    output_folder_path = 'D:/Code/code/submit/bms_ai/PycharmProjects/data/filter_data/badcase_filter/'  # 替换为你想要保存处理后文件的文件夹路径

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    i = 0
    for filename in tqdm(os.listdir(input_folder_path)):
        if (i >-1):
            i = i + 1

            file_path = os.path.join(input_folder_path, filename)
            if '.csv' in filename:
                # 读取CSV文件
                df = pd.read_csv(file_path,on_bad_lines='warn')
            elif '.xlsx' in filename:
                df = pd.read_excel(file_path)
                filename = filename.replace('.xlsx', '.csv')
            else:
                continue

            # 处理数据
            processed_df= df.sort_values('businesstimestamp', ignore_index=True)
            # print(filename)
            processed_df  = data_prepare(processed_df)
            # 构造输出文件名
            output_filename = f"processed_{filename}"
            output_file_path = os.path.join(output_folder_path, output_filename)

            # 保存处理后的数据到新文件
            if (len(processed_df) > 0):
                processed_df.to_csv(output_file_path, index=False)
                print(f"Processed and saved {filename} to {output_filename}")
            else:
                print(filename)

if __name__ == "__main__":
    main()