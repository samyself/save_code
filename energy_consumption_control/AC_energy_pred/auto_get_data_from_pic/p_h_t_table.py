import copy
import pickle

import numpy as np

if __name__ == '__main__':
    pkl_path = './pic_data_dict.pkl'
    table_pkl_path = './p_h_t_table_dict.pkl'
    with open(pkl_path, 'rb') as f:
        all_data_dict = pickle.load(f)

    all_p = []
    all_t = []
    all_h = []
    # all_x = []
    for k in all_data_dict:
        data = all_data_dict[k]

        for para_name in data:
            if 'T' in para_name:
                if data[para_name] == "NA":
                    break
                temp = float(data[para_name])
                all_t.append(temp)
            if 'p' in para_name:
                pressure = float(data[para_name])
                all_p.append(pressure)

            # if 'x' in para_name:
            #     x = float(data[para_name])
            #     all_x.append(x)

            if 'h' in para_name:
                h = float(data[para_name])
                all_h.append(h)

    all_p = np.array(all_p)
    all_t = np.array(all_t)
    # all_x = np.array(all_x)
    all_h = np.array(all_h)

    unique_p = np.sort(np.unique(all_p))
    unique_t = np.sort(np.unique(all_t))
    unique_h = np.sort(np.unique(all_h))

    t_value = []
    for p_index in range(len(unique_p)):
        if p_index==69:
            continue
        now_p = unique_p[p_index]
        p_place = (all_p == now_p)
        this_t = copy.deepcopy(all_t[p_place])
        # this_x = copy.deepcopy(all_x[p_place])
        this_h = copy.deepcopy(all_h[p_place])

        rank = np.argsort(this_h)
        this_ranked_h = this_h[rank]
        this_ranked_t = this_t[rank]

        # 后处理没有识别负号的加上负号
        for t_index in range(1, len(this_ranked_t)):
            now_data = this_ranked_t[t_index]
            last_data = this_ranked_t[t_index - 1]
            if last_data < 0 and now_data >= 0:
                diff = now_data - last_data
                max_diff = 8
                if diff > max_diff:
                    this_ranked_t[t_index] = -this_ranked_t[t_index]

        t_value.append(this_ranked_t)

    log_p = np.log(unique_p)[:-1]
    t_value = np.array(t_value)
    table_dict = {
        'log_p': log_p,
        'h': unique_h,
        't': t_value
    }
    with open(table_pkl_path, 'wb') as f:
        pickle.dump(table_dict, f)
