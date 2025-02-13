import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from scipy.interpolate import interp2d, interp1d
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from dataset.wzl_dataset_wzl import eval_dataset_soc
# from fusion_model import Fusion_model as my_model
# from transformer_model import Transformer as my_model
from model.wzl_lstm_model import LSTMModel as my_model
import time
from tqdm import tqdm
import pickle


    
def eval_net(net, eval_dataloader, ckpt_path, device):
    all_info = eval_dataloader.dataset.all_info
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    diff = []
    bms_diff = []
    pred = []
    gt = []
    bms_soc = []
    len_all = 0
    len_max_3 = 0
    len_bms_max_3 = 0
    #------------------
    max_90_100_2 = 0
    max_90_100_3 = 0
    max_80_90_3 = 0
    max_80_90_5 = 0
    #------------------
    net.eval()
    for batch in tqdm(eval_dataloader):
        input_data, y = batch
        input_data = input_data.to(device).reshape(-1, 300, 7)
        y = y.to(device).reshape(-1, 2, 2)
        split_num = input_data.shape[0]

        labels = y[:, :, 0].to(device).reshape(-1, 1)
        bms = y[:, :, 1].to(device).reshape(-1, 1)
        inputs = input_data.to(device).reshape(-1, 300, 7)
        
        with torch.no_grad():
            # outputs = net(inputs.to(dtype=torch.float32)).to("cuda")[:, 0, :].reshape(-1, 1)
            outputs= net(inputs.to(dtype=torch.float32)).to("cpu").reshape(-1, 1)
        diff.append(torch.abs(outputs-labels).cpu())
        bms_diff.append(torch.abs(bms-labels).cpu())
        pred.append(outputs.cpu())
        gt.append(labels.cpu())
        bms_soc.append(bms.cpu())  
        if (torch.abs(outputs-labels).cpu() > 3).any():
            len_max_3 += 1
        if (torch.abs(bms-labels).cpu() > 3).any():
            len_bms_max_3 += 1
        len_all += 1
        #--------------------------------------------------
        # range_80_90 = (labels >= 80) & (labels < 85)
        # range_90_100 = (labels >= 90) & (labels < 100)
        range_80_90 = (bms >= 80) & (bms < 85)
        range_90_100 = (bms >= 90) & (bms < 100)
        out_80_90 = outputs[range_80_90]
        out_90_100 = outputs[range_90_100]
        lab_80_90 = labels[range_80_90]
        lab_90_100 = labels[range_90_100]
        #----------------------------------------------------
        if (torch.abs(out_90_100 - lab_90_100).cpu() > 2).any():
            max_90_100_2 += 1
        if (torch.abs(out_90_100 - lab_90_100).cpu() > 3).any():
            max_90_100_3 += 1
        if (torch.abs(out_80_90 - lab_80_90).cpu() > 3).any():
            max_80_90_3 += 1
        if (torch.abs(out_80_90 - lab_80_90).cpu() > 5).any():
            max_80_90_5 += 1
    print(f"90-100最大误差超过2的工况数：{max_90_100_2}")
    print(f"90-100最大误差超过3的工况数：{max_90_100_3}")
    print(f"80-90最大误差超过3的工况数：{max_80_90_3}")
    print(f"80-90最大误差超过5的工况数：{max_80_90_5}")
    print("-"*30)
    print(f"工况总数：{len_all}")
    print(f"模型最大误差超过3的工况个数:{len_max_3}")
    print(f"BMS最大误差超过3的工况个数:{len_bms_max_3}")
    print("-"*30)
    res = np.concatenate(diff)
    bms_res = np.concatenate(bms_diff)
    fenlei = np.concatenate(gt)
    
    range_80_85 = (fenlei >= 80) & (fenlei < 85)
    range_85_90 = (fenlei >= 85) & (fenlei < 90)
    range_90_95 = (fenlei >= 90) & (fenlei < 95)
    range_95_100 = (fenlei >= 95) & (fenlei <= 100)
    
    res_80_85 = res[range_80_85]
    res_85_90 = res[range_85_90]
    res_90_95 = res[range_90_95]
    res_95_100 = res[range_95_100]
    
    bms_res_80_85 = bms_res[range_80_85]
    bms_res_85_90 = bms_res[range_85_90]
    bms_res_90_95 = bms_res[range_90_95]
    bms_res_95_100 = bms_res[range_95_100]
    #-----------------------------------------------------------
    keys = [f'arr_{i}' for i in range(len(pred))]
    np.savez('../data/soc/res/pred.npz', **dict(zip(keys, pred)))

    keys = [f'arr_{i}' for i in range(len(gt))]
    np.savez('../data/soc/res/gt.npz', **dict(zip(keys, gt)))

    keys = [f'arr_{i}' for i in range(len(bms_soc))]
    np.savez('../data/soc/res/bms.npz', **dict(zip(keys, bms_soc)))

    with open('../data/soc/res/name.pkl', 'wb') as file:
        pickle.dump(all_info, file)

    # np.save('/home/workspace/soc/res/w.npy', w.cpu().detach().numpy())
    #---------------------------------------------------------
    print("-"*30)
    print(f"总长度：{res.shape[0]}秒")
    print("-"*30)
    print(f"模型的MAE:{np.mean(res)}")
    print(f"模型最大误差:{np.max(res)}")
    print(f"BMS的MAE:{np.mean(bms_res)}")
    print(f"BMS的最大误差:{np.max(bms_res)}")
    print("-"*30)
    print(f"80到85的秒数:{res_80_85.shape[0]}")
    print(f"85到90的秒数:{res_85_90.shape[0]}")
    print(f"90到95的秒数:{res_90_95.shape[0]}")
    print(f"95到100的秒数:{res_95_100.shape[0]}")
    print("-"*30)
    print(f"80到85_模型的MAE:{np.mean(res_80_85)}")
    print(f"85到90_模型的MAE:{np.mean(res_85_90)}")
    print(f"90到95_模型的MAE:{np.mean(res_90_95)}")
    print(f"95到100_模型的MAE:{np.mean(res_95_100)}")
    print("-"*30)
    print(f"80到85_模型的max_error:{np.max(res_80_85)}")
    print(f"85到90_模型的max_error:{np.max(res_85_90)}")
    print(f"90到95_模型的max_error:{np.max(res_90_95)}")
    print(f"95到100_模型的max_error:{np.max(res_95_100)}")
    print("-"*30)
    print(f"80到85_BMS的MAE:{np.mean(bms_res_80_85)}")
    print(f"85到90_BMS的MAE:{np.mean(bms_res_85_90)}")
    print(f"90到95_BMS的MAE:{np.mean(bms_res_90_95)}")
    print(f"95到100_BMS的MAE:{np.mean(bms_res_95_100)}")
    print("-"*30)
    
    

if __name__ == "__main__":
    device = torch.device("cpu")
    data_folder = '../data/filter_data/data_handle_new'
    # data_folder = '/home/workspace/soc/data_demo'
    all_file_name = os.listdir(data_folder)
    all_data_path = [os.path.join(data_folder, file_name) for file_name in all_file_name]
    seed = 111
    train_eval_rate = 0.5
    np.random.seed(seed)
    all_data_index = np.arange(len(all_data_path))
    np.random.shuffle(all_data_index)
    torch.manual_seed(seed)

    train_final_place = int(train_eval_rate * len(all_data_index))
    train_index = all_data_index[:train_final_place]
    eval_index = all_data_index[train_final_place:]

    train_data_path = [all_data_path[i] for i in train_index]
    eval_data_path = [all_data_path[i] for i in eval_index]

    eval_dataset = eval_dataset_soc(eval_data_path)
    #---------------------------------------------------------------------------------------------------
    net = my_model().to(device)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=10)
    ckpt_path = "../data/ckpt/wzl_model.pth"
    eval_net(net = net, eval_dataloader = eval_dataloader, ckpt_path = ckpt_path, device=device)