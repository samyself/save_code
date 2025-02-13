import pandas as pd
import sys

sys.path.append('/home/work/meiyuheng_Project/bms_ai/PycharmProjects/pythonProject2')
import os

print(os.getcwd())
from PycharmProjects.pythonProject2.common_func import draw_pred, dictlist_to_listdict, create_histogram
import numpy as np
import json
from tqdm import tqdm
import threading



if __name__ == '__main__':
    intput_csv_folder = '../../data/filter_data/data_handle_new'
    all_path_list = os.listdir(intput_csv_folder)
    print(all_path_list[0])

    all_info_list = []
    # Curr_all = []
    Curr_80_85 = np.array([])
    Curr_85_90 = np.array([])
    Curr_90_95 = np.array([])
    Curr_95_100 = np.array([])

    print('len(all_path_list)', len(all_path_list))
    for i in tqdm(range(len(all_path_list))):
        path = all_path_list[i]
        if '.csv' in path:
            pass
        else:
            continue
        path = os.path.join(intput_csv_folder, path)
        df = pd.read_csv(path)

        Curr = df['Curr']
        BmsSoc = df['BmsSoc']
        mask1 = (BmsSoc >= 80) & (BmsSoc < 85)
        mask2 = (BmsSoc >= 85) & (BmsSoc < 90)
        mask3 = (BmsSoc >= 90) & (BmsSoc < 95)
        mask4 = (BmsSoc >= 95) & (BmsSoc <= 100)
        Curr_80_85_split = np.array(Curr[mask1])
        Curr_85_90_split = np.array(Curr[mask2])
        Curr_90_95_split = np.array(Curr[mask3])
        Curr_95_100_split = np.array(Curr[mask4])

        Curr_80_85 = np.append(Curr_80_85, Curr_80_85_split)
        Curr_85_90 = np.append(Curr_85_90, Curr_85_90_split)
        Curr_90_95 = np.append(Curr_90_95, Curr_90_95_split)
        Curr_95_100 = np.append(Curr_95_100, Curr_95_100_split)


    Curr_all = [Curr_80_85, Curr_85_90, Curr_90_95, Curr_95_100]
    print('Curr_all_len', len(Curr_all))

    base_curr = 80
    for i in range(len(Curr_all)):
        print(f'{base_curr} ~ {base_curr+5}:')
        create_histogram(Curr_all[i], 10)
        base_curr += 5

