import copy
import pickle

import numpy as np

if __name__ == '__main__':
    pkl_path = './pic_data_dict.pkl'
    table_pkl_path = './p_h_x_table_dict.pkl'
    with open(pkl_path, 'rb') as f:
        all_data_dict = pickle.load(f)

    all_p = []
    all_t = []
    all_h = []
    all_x = []
    for k in all_data_dict:
        data = all_data_dict[k]

        for para_name in data:
            if 'T' in para_name:
                temp = float(data[para_name])
                all_t.append(temp)
            if 'p' in para_name:
                pressure = float(data[para_name])
                all_p.append(pressure)

            if 'x' in para_name:
                x = float(data[para_name])
                all_x.append(x)

            if 'h' in para_name:
                h = float(data[para_name])
                all_h.append(h)

    all_p = np.array(all_p)
    all_t = np.array(all_t)
    all_x = np.array(all_x)
    all_h = np.array(all_h)

    unique_p = np.sort(np.unique(all_p))
    unique_t = np.sort(np.unique(all_t))
    unique_h = np.sort(np.unique(all_h))

    x_value = []
    for p_index in range(len(unique_p)):
        now_p = unique_p[p_index]
        p_place = (all_p == now_p)
        this_t = copy.deepcopy(all_t[p_place])
        this_x = copy.deepcopy(all_x[p_place])
        this_h = copy.deepcopy(all_h[p_place])

        rank = np.argsort(this_h)
        this_ranked_h = this_h[rank]
        this_ranked_x = this_x[rank]

        # 不在h里面的补全 如果小于 则按照0 大于按照1
        this_min_h = min(this_ranked_h)
        zero_num = len(unique_h[unique_h < this_min_h])

        this_max_h = max(this_ranked_h)
        one_num = len(unique_h[unique_h > this_max_h])

        this_x = np.append(this_ranked_x, np.ones(one_num))
        this_x = np.append(np.zeros(zero_num), this_x)

        x_value.append(this_x)

    log_p = np.log(unique_p)
    x_value = np.array(x_value)
    table_dict = {
        'log_p': log_p,
        'h': unique_h,
        'x': x_value
    }
    with open(table_pkl_path, 'wb') as f:
        pickle.dump(table_dict, f)
