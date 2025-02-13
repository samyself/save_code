import argparse
import os.path
import pickle

import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from tqdm import tqdm

import torch.nn.functional as F

import torch.nn as nn
import matplotlib.pyplot as plt
import config_all as config
from data_utils import get_device
from dataset.dataset_LoSideP import dataset_LoSideP
from draw_pic import draw_pred
from model.model_LoSideP import MLPModel

def train_net(net, train_epoch, train_loader, optimizer, device):
    criterion = nn.MSELoss()

    iterator = tqdm(range(train_epoch))
    for epoch_index in iterator:
        for batch_idx, (input_data, y) in enumerate(train_loader):
            #转化为tensor
            input_data = torch.as_tensor(input_data, dtype=torch.float32)
            y = torch.as_tensor(y, dtype=torch.float32)

            #按样本进行数据归一化
            input_data = F.normalize(input_data, p=2, dim=1)
            input_data, y = input_data.to(device), y.to(device)

            #前向传播 
            pred = net(input_data)
            

            loss_1 = criterion(pred[:, 0], y[:, 0])
            loss_2 = criterion(pred[:, 1], y[:, 1])
            loss = loss_1 + loss_2

            optimizer.zero_grad()
            loss.backward()
            iterator.desc = f"epoch_index:{epoch_index},loss:{loss.item()}"

            #限制梯度大小
            # scale = config.gard_scale
            # for name, para in net.named_parameters():
            #     if para.any().grad:
            #         scaled_grad = scale * para.data * torch.sgn(para.grad)

            #         # para.grad = scaled_grad
            #         if torch.abs(scaled_grad) < torch.abs(para.grad):
            #             para.grad = scaled_grad

            optimizer.step()
    return

def eval_net(net, eval_loader, result_pic_folder, device):
    net.eval()
    all_TmRteOut_LoSideP_mae = []
    all_lo_pressure_temp_mae = []
    all_info = eval_loader.dataset.all_info
    with torch.no_grad():
        # 遍历工况
        batch_idx = 0
        for batch in tqdm(eval_loader):
            input_data, y = batch
            # print(input_data, y)
            # exit()
            split_num = len(input_data)
            for split_index in range(split_num):
                # if '2023-11-30 16_23_06_MS11_THEM_A_E4U3_T151, V015_Amb-7~-15℃_2人_后排关闭_40kph_行车加热_座椅加热(无感)_能耗测试' in all_info[batch_idx][split_index]:
                #     print()

                split_input = input_data[split_index][0].to(device)
                split_y = y[split_index][0].to(device)

                # split_len = len(split_input)
                # if split_len < 50:
                #     continue
                pred = net(split_input)

                # refrigerant_mix_temp_mae = torch.mean(torch.abs(pred_t_r_1 - refrigerant_mix_temp))
                TmRteOut_LoSideP_mae = torch.mean(torch.abs(pred[:, 0] - split_input[:, 0]))
                lo_pressure_temp_mae = torch.mean(torch.abs(pred[:, 1] - split_input[:, 1]))

                # all_refrigerant_mix_temp_mae.append(refrigerant_mix_temp_mae.view(1, -1))
                all_TmRteOut_LoSideP_mae.append(TmRteOut_LoSideP_mae.view(1, -1))
                all_lo_pressure_temp_mae.append(lo_pressure_temp_mae.view(1, -1))

                # 可视化
                series_list = [
                    # [pred_t_r_1, refrigerant_mix_temp],
                    [pred[:, 0], split_input[:, 0]],
                    [pred[:, 1], split_input[:, 1]],
                ]
                series_name_list = [
                    ['pred_TmRteOut_LoSideP', 'TmRteOut_LoSideP'],
                    ['pred_lo_pressure_temp', 'lo_pressure_temp'],
                ]
                file_name = all_info[batch_idx][split_index].replace('.csv', '').split('/')[-1].split('\\')[-1]
                pic_name = file_name + '.png'
                draw_pred(series_list, series_name_list, result_pic_folder, pic_name)

            batch_idx += 1

    # 计算指标
    # all_refrigerant_mix_temp_mae = torch.cat(all_refrigerant_mix_temp_mae)
    all_TmRteOut_LoSideP_mae = torch.cat(all_TmRteOut_LoSideP_mae)
    all_lo_pressure_temp_mae = torch.cat(all_lo_pressure_temp_mae)

    # mean_refrigerant_mix_temp_mae = torch.mean(all_refrigerant_mix_temp_mae)
    mean_TmRteOut_LoSideP_mae = torch.mean(all_TmRteOut_LoSideP_mae)
    mean_lo_pressure_temp_mae = torch.mean(all_lo_pressure_temp_mae)
    return mean_TmRteOut_LoSideP_mae, mean_lo_pressure_temp_mae
# -----------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default='2', type=str, help='GPU ids')
    args = parser.parse_args()
    device = get_device(device_id=args.device_id)

    device = torch.device("cpu")
    data_folder = "D:\\energy_consumption_control-main\\AC_energy_pred\\data\\csv_all"
    all_file_name = os.listdir(data_folder)
    all_data_path = [os.path.join(data_folder, file_name) for file_name in all_file_name]

    result_pic_folder = './data/train_result/pic'

    # 模型路径
    ckpt_folder = './data/ckpt/'
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
        
    # 划分train eval
    seed = 111
    train_eval_rate = 0.6
    np.random.seed(seed)
    all_data_index = np.arange(len(all_data_path))
    np.random.shuffle(all_data_index)
    torch.manual_seed(seed)

    train_final_place = int(train_eval_rate * len(all_data_index))
    train_index = all_data_index[:train_final_place]
    eval_index = all_data_index[train_final_place:]

    train_data_path = [all_data_path[i] for i in train_index]
    eval_data_path = [all_data_path[i] for i in eval_index]

    load_dataset = False
    train_dataset_path = './data/train_dataset.pkl'
    eval_dataset_path = './data/eval_dataset.pkl'
    all_dataset_path = './data/all_dataset.pkl'

    if load_dataset:
        with open(train_dataset_path, 'rb') as f:
            train_dataset = pickle.load(f)

        with open(eval_dataset_path, 'rb') as f:
            eval_dataset = pickle.load(f)

        with open(all_dataset_path, 'rb') as f:
            all_dataset = pickle.load(f)
    else:
        train_dataset = dataset_LoSideP(train_data_path, True)
        eval_dataset = dataset_LoSideP(eval_data_path, False)
        # all_dataset = dataset_LoSideP(all_data_path, True)

        with open(train_dataset_path, 'wb') as f:
            pickle.dump(train_dataset, f)

        with open(eval_dataset_path, 'wb') as f:
            pickle.dump(eval_dataset, f)

        # with open(all_dataset_path, 'wb') as f:
        #     pickle.dump(all_dataset, f)

    # 网络
    input_dim = 7
    net = MLPModel(input_dim) #实例化

    load_pretrain = False
    pretrain_ckpt_path = ""
    if load_pretrain:
        if os.path.exists(pretrain_ckpt_path):
            net.load_state_dict(torch.load(pretrain_ckpt_path, map_location=torch.device('cpu')))
    else:
        net.initialize() #权重初始化
    net.to(device)

    # 训练
    # lr = config.lr
    lr = 1e-3
    # train_batch_size = config.train_batch_size
    train_batch_size = 512
    train_epoch = config.train_epoch
    optimizer = optim.SGD(net.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    train_net(net, train_epoch, train_loader, optimizer, device)

    #保存模型
    saved_model_name = "LoSideP_model.pth"
    saved_ckpt_path = os.path.join(ckpt_folder, saved_model_name)
    torch.save(net.state_dict(), saved_ckpt_path)

    #测试
    eval_batch_size = config.eval_batch_size
    cond_num = len(eval_dataset)
    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
    mean_TmRteOut_LoSideP_mae, mean_TmRteOut_LoSideP_mae = eval_net(net, eval_loader, result_pic_folder, device)
    print(mean_TmRteOut_LoSideP_mae, mean_TmRteOut_LoSideP_mae)

