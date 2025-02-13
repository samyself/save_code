import argparse
import os.path
import pickle

import torch
from pyexpat import features
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from tqdm import tqdm


import torch.nn as nn

import config
from data_utils import get_device
from dataset.dataset_cmpr_control import CmprControlBaseDataset
from draw_pic import draw_pred
# from model.model_cmpr_control_tcn import *
from model.model_cmpr_control2_v3_b1 import *
from model.my_common import get_loss_cur
'''
压缩机控制模型
'''
lr = 1e-2
train_batch_size = 1
weight_decay = 0.01
train_epoch = 20
load_pretrain = False
pretrain_ckpt_path = './data/ckpt/ccsingl2_0924_v1.pth'
saved_ckpt_path  = 'data/ckpt/ccsingl2_0924_v1.pth'
Flag = 2
Flag_train = True
# saved_ckpt_path = './data/ckpt/cmpr_control_net_v1.pth'
result_pic_folder = './data/train_result/pic/submits10924_v1'


def custom_loss(output, target, max_val, min_val):
    # Sloss = nn.SmoothL1Loss(beta=1000)
    Sloss = nn.L1Loss(reduction='mean')
    mse_loss = Sloss(output,target)
    total_loss = mse_loss
    return total_loss


def train_net(net, train_epoch, train_loader,eval_loader, optimizer, device):
    criterion = custom_loss
    # criterion = nn.MSELoss(reduction='mean')
    train_loss = []
    eval_loss = []
    iterator = tqdm(range(train_epoch))
    # 使用 StepLR 调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    A = 20
    for epoch_index in iterator:
        loss_list = []
        for batch_idx, (input_data, y) in enumerate(train_loader):
            input_data, y = input_data.to(device), y.to(device)

            compressor_speed_real = y[:,0]
            pred_output = net(input_data)
            cab_heating_status_act_pos_real = y[:,1]

            compressor_speed_pred = pred_output[0]
            cab_heating_status_act_pos_pred = pred_output[1]

            loss_compressor_speed =  criterion(compressor_speed_pred, compressor_speed_real,net.compressor_speed_max,net.compressor_speed_min)
            loss_act_pos =  criterion(cab_heating_status_act_pos_pred, cab_heating_status_act_pos_real,net.cab_heating_status_act_pos_max,net.cab_heating_status_act_pos_min)

            if Flag == 1:
                loss = loss_compressor_speed
            elif Flag == 2:
                loss = loss_act_pos
            else:
                loss = loss_act_pos+loss_compressor_speed
            # print('loss_compressor_speed',loss_compressor_speed,'loss_act_pos',loss_act_pos)


            optimizer.zero_grad()
            loss.backward()
            # iterator.desc = f"epoch_index:{epoch_index},loss:{loss.item()}"
            loss_list.append(loss.item())
            # 限制梯度大小

            # scale = config.gard_scale
            # for name, para in net.named_parameters():
            #     if torch.any(para.grad !=0):
            #         scaled_grad = scale * para.data * torch.sgn(para.grad)
            #         # 创建一个布尔tensor，当scaled_grad的绝对值小于para.grad的绝对值时为True
            #         condition = torch.abs(scaled_grad) < torch.abs(para.grad)
            #         # 使用这个布尔tensor来选择性地更新梯度
            #         para.grad = torch.where(condition, scaled_grad, para.grad)

            optimizer.step()
            # with torch.no_grad():
            #     net.cab_pos_n.clamp_(min=1.0, max=2.0)
        scheduler.step()
        train_loss.append(sum(loss_list)/len(loss_list))
        # eval_loss.append(eavl_loss(net,eval_loader,device))
        eval_loss = [0]
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch [{epoch_index + 1}/{train_epoch}],Train Loss: {train_loss[-1]:.4f},eval Loss: {eval_loss[-1]:.4f}, Learning Rate: {current_lr:.6f}')

        if train_loss[-1]<A:
            A = train_loss[-1]
            torch.save(net, saved_ckpt_path)
    print('min loss=',A)
    return train_loss,eval_loss


def eval_net(net, eval_loader, result_pic_folder, device):
    net.eval()
    # all_refrigerant_mix_temp_mae = []
    # all_refrigerant_after_heat_exchange_mae = []
    # all_air_temp_mae = []
    all_info = eval_loader.dataset.all_info
    all_compressor_speed_mae = []
    all_cab_heating_status_act_pos_mae = []
    all_compressor_speed_prb_mae = []
    all_cab_heating_status_act_pos_prb_mae = []
    with torch.no_grad():
        # 遍历工况
        batch_idx = 0
        for batch in tqdm(eval_loader):
            input_data, y = batch
            split_num = len(input_data)

            for split_index in range(split_num):
                # if '2023-11-30 16_23_06_MS11_THEM_A_E4U3_T151, V015_Amb-7~-15℃_2人_后排关闭_40kph_行车加热_座椅加热(无感)_能耗测试' in all_info[batch_idx][split_index]:
                #     print()
                split_input = input_data[split_index][0].to(device)
                split_y = y[split_index][0].to(device)
                compressor_speed_pred = []
                cab_heating_status_act_pos_pred = []
                for index,(input,y) in enumerate(zip(split_input,split_y)):
                    pred_output = net(input)
                    compressor_speed_pred_1 = pred_output[0].tolist()
                    cab_heating_status_act_pos_pred_1 = pred_output[1].tolist()

                    compressor_speed_pred.append(compressor_speed_pred_1)
                    cab_heating_status_act_pos_pred.append(cab_heating_status_act_pos_pred_1)

                compressor_speed_pred = torch.tensor(compressor_speed_pred)
                cab_heating_status_act_pos_pred = torch.tensor(cab_heating_status_act_pos_pred)
                compressor_speed_real = split_y[:, 0]
                cab_heating_status_act_pos_real = split_y[:, 1]
                compressor_speed_mae = torch.mean(torch.abs(compressor_speed_pred - compressor_speed_real))
                compressor_speed_mae_prob = torch.abs(compressor_speed_pred - compressor_speed_real) / compressor_speed_real
                cab_heating_status_act_pos_mae = torch.mean(torch.abs(cab_heating_status_act_pos_pred - cab_heating_status_act_pos_real))
                cab_heating_status_act_pos_mae_prob = torch.abs(cab_heating_status_act_pos_pred - cab_heating_status_act_pos_real) / cab_heating_status_act_pos_real
                # 可视化
                series_list = [
                    # [pred_t_r_1, refrigerant_mix_temp],
                    [compressor_speed_pred, compressor_speed_real],
                    [cab_heating_status_act_pos_pred, cab_heating_status_act_pos_real],
                    [compressor_speed_mae_prob, cab_heating_status_act_pos_mae_prob]
                ]
                series_name_list = [
                    # ['pred_refrigerant_mix_temp', 'refrigerant_mix_temp'],
                    ['pred_compressor_speed', 'compressor_speed'],
                    ['pred_cab_heating_status_act_pos', 'cab_heating_status_act_pos'],
                    ['pred_compressor_speed_prob','pred_cab_heating_status_act_pos_prob' ]
                ]
                file_name = all_info[batch_idx][split_index].replace('.csv', '').split('/')[-1].split('\\')[-1]
                pic_name = file_name + '.png'
                # print(series_list)
                draw_pred(series_list, series_name_list, result_pic_folder, pic_name)
                all_compressor_speed_mae.append(compressor_speed_mae.view(1, -1))
                all_cab_heating_status_act_pos_mae.append(cab_heating_status_act_pos_mae.view(1, -1))
                all_compressor_speed_prb_mae.append(compressor_speed_mae_prob.view(1, -1))
                all_cab_heating_status_act_pos_prb_mae.append(cab_heating_status_act_pos_mae_prob.view(1, -1))

            batch_idx += 1

    net.train()

    # 计算指标
    # all_refrigerant_mix_temp_mae = torch.cat(all_refrigerant_mix_temp_mae)
    all_compressor_speed_mae = torch.cat(all_compressor_speed_mae)
    all_cab_heating_status_act_pos_mae = torch.cat(all_cab_heating_status_act_pos_mae)
    all_compressor_speed_prb_mae = torch.cat(all_compressor_speed_prb_mae)
    all_cab_heating_status_act_pos_prb_mae = torch.cat(all_cab_heating_status_act_pos_prb_mae)

    # mean_refrigerant_mix_temp_mae = torch.mean(all_refrigerant_mix_temp_mae)
    mae_all_compressor_speed_mae = torch.mean(all_compressor_speed_mae)
    mae_all_cab_heating_status_act_pos_mae = torch.mean(all_cab_heating_status_act_pos_mae)
    mae_all_compressor_speed_prb_mae = torch.mean(all_compressor_speed_prb_mae)
    mae_all_cab_heating_status_act_pos_prb_mae = torch.mean(all_cab_heating_status_act_pos_prb_mae)

    return mae_all_compressor_speed_mae, mae_all_cab_heating_status_act_pos_mae,mae_all_compressor_speed_prb_mae,mae_all_cab_heating_status_act_pos_prb_mae


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default='2', type=str, help='GPU ids')
    args = parser.parse_args()
    device = get_device(device_id=args.device_id)


    if not os.path.exists(result_pic_folder):
        os.makedirs(result_pic_folder)

    data_folder = './data/csv'
    all_file_name = os.listdir(data_folder)
    all_data_path = [os.path.join(data_folder, file_name) for file_name in all_file_name]

    # 模型路径
    ckpt_folder = './data/ckpt/'
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    # 划分train eval
    seed = 2
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

    load_dataset = True
    train_dataset_path = './data/cmpr_control_train_dataset_full_seed2.pkl'
    eval_dataset_path = './data/cmpr_control_eval_dataset_full_seed2.pkl'
    all_dataset_path = './data/cmpr_control_all_dataset_full_seed2.pkl'

    if load_dataset:
        with open(train_dataset_path, 'rb') as f:
            train_dataset = pickle.load(f)

        with open(eval_dataset_path, 'rb') as f:
            eval_dataset = pickle.load(f)

        with open(all_dataset_path, 'rb') as f:
            all_dataset = pickle.load(f)
    else:
        train_dataset = CmprControlBaseDataset(train_data_path, True)
        eval_dataset = CmprControlBaseDataset(eval_data_path, False)
        all_dataset = CmprControlBaseDataset(all_data_path, False)


        with open(train_dataset_path, 'wb') as f:
            pickle.dump(train_dataset, f)

        with open(eval_dataset_path, 'wb') as f:
            pickle.dump(eval_dataset, f)

        with open(all_dataset_path, 'wb') as f:
            pickle.dump(all_dataset, f)



    # 网络

    input_dim = 9
    output_dim = 2
    net = MLPModel(input_dim=6, output_dim=1)

    if load_pretrain:
        if os.path.exists(pretrain_ckpt_path):
            net=torch.load(pretrain_ckpt_path, map_location=torch.device('cpu'))
            # net.load_state_dict(torch.load(pretrain_ckpt_path, map_location=torch.device('cpu')))
    net.to(device)

    # 训练
    # lr = config.lr
    # train_batch_size = config.train_batch_size
    # weight_decay = 0.01
    # train_epoch = config.train_epoch
    # optimizer = optim.SGD(net.parameters(), lr=lr)
    # 这里使用的是Adam优化算法


    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False)
    if Flag_train:
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)
        train_loss_list,eavl_loss_list = train_net(net, train_epoch, train_loader,eval_loader, optimizer, device)

        get_loss_cur(train_loss_list,eavl_loss_list,['train loss','eval loss'])
        # torch.save(net, saved_ckpt_path)
        # print(f"save success {saved_ckpt_path}")

    # 测试    eval_batch_size = config.eval_batch_size
    cond_num = len(eval_dataset)


    # eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
    mae_all_compressor_speed_mae, mae_all_cab_heating_status_act_pos_mae, all_compressor_speed_prb_mae,all_cab_heating_status_act_pos_prb_mae = eval_net(net, eval_loader, result_pic_folder, device)

    # print('mean_refrigerant_mix_temp_mae', mean_refrigerant_mix_temp_mae)
    print('mae_all_compressor_speed_mae', mae_all_compressor_speed_mae)
    print('mae_all_cab_heating_status_act_pos_mae', mae_all_cab_heating_status_act_pos_mae)
    print('all_compressor_speed_prb_mae', all_compressor_speed_prb_mae)
    print('all_cab_heating_status_act_pos_prb_mae', all_cab_heating_status_act_pos_prb_mae)
