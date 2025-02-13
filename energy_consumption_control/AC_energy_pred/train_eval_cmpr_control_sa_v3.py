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
from model.model_cmpr_control2_v2 import *
from model.my_common import get_loss_cur
'''
压缩机控制模型
'''
lr = 1e-5
train_batch_size = config.train_batch_size
weight_decay = 0.01
train_epoch = 20
load_pretrain = True
pretrain_ckpt_path = './data/ckpt/cc_0926_v2.pth'
saved_ckpt_path  = 'data/ckpt/cc_0926_v2_1.pth'
Flag = 3
Flag_train = True
# saved_ckpt_path = './data/ckpt/cmpr_control_net_v1.pth'
result_pic_folder = './data/train_result/pic/submits10926_v1'
best_loss = 400

"""
    output: batchsize*n(输出)
    target: batchsize*n(真实值)
    flag: 1-compressor_speed,2-cab_heating_status_act_po,3=1+2
    func: 损失函数
    return: 总loss
"""
def custom_loss(output, target,flag):
    Sloss = nn.L1Loss(reduction='mean')
    # Sloss = nn.SmoothL1Loss(beta=1000)
    if flag == 1 or flag == 2:
        total_loss = Sloss(output[:,flag-1],target[:,flag-1])
    else:
        total_loss = torch.tensor(0.0)
        for i in range(flag-1):
            loss_temp = Sloss(output[:,i],target[:,i])
            total_loss = total_loss + loss_temp

    return total_loss

def get_data_loss(net,train_loader,flag):
    net.eval()
    with torch.no_grad():
        loss_list = []
        for batch_idx, (input_data, y) in enumerate(train_loader):
            input_data, y = input_data.to(device), y.to(device)
            loss = custom_loss(net(input_data), y, flag)
            loss_list.append(loss.item())
    net.train()
    return sum(loss_list)/len(loss_list)
"""


    weights:当前网络参数
    func:随机更新网络参数
    return: 返回优化后的weights
"""
def perturb_weights(weights):
    new_weights = {}
    for key, value in weights.items():
        if value.dtype != torch.float32:
            continue
        noise = torch.randn_like(value) * lr  # 小幅度扰动
        new_weights[key] = value + noise
    return new_weights


"""
"""
def accept_worse_solution(current_loss, new_loss, temperature):
    if new_loss < current_loss:
        return True
    else:
        acceptance_prob = np.exp(-abs(current_loss - new_loss) / temperature)
        return np.random.rand() < acceptance_prob

def train_net(net, train_epoch, train_loader,eval_loader, optimizer, device,temperature=1.0):
    criterion = custom_loss
    current_weights = net.state_dict()
    cooling_rate = 0.99
    # criterion = nn.MSELoss(reduction='mean')
    train_loss = []
    eval_loss = []
    iterator = tqdm(range(train_epoch))
    # 使用 StepLR 调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    A = best_loss
    for epoch_index in iterator:
        loss_list = []
        for batch_idx, (input_data, y) in enumerate(train_loader):
            input_data, y = input_data.to(device), y.to(device)
            pred_output = net(input_data)
            loss = criterion(pred_output, y,Flag)

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

        current_loss = get_data_loss(net,train_loader,Flag)
        current_weights = net.state_dict()
        # 生成新的权重配置
        new_weights = perturb_weights(current_weights)
        net.load_state_dict(new_weights)
        new_loss = get_data_loss(net,train_loader,Flag)

        # 如果效果好就接受新解，否则就还原为原来的解
        if accept_worse_solution(current_loss, new_loss, temperature):
           pass
        else:
            net.load_state_dict(current_weights)
            new_loss = current_loss

        # 降温
        temperature *= cooling_rate
        if temperature < 1e-6:
            break

        # scheduler.step()
        train_loss.append(new_loss)
        eval_loss.append(eavl_loss(net,eval_loader,device))
        # eval_loss = [0]
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch [{epoch_index + 1}/{train_epoch}],Train Loss: {train_loss[-1]:.4f},eval Loss: {eval_loss[-1]:.4f}, Learning Rate: {current_lr:.6f}')

        if eval_loss[-1]<A:
            A = eval_loss[-1]
            torch.save(net, saved_ckpt_path)
    print('min loss=',A)
    return train_loss,eval_loss

def eavl_loss(net, eval_loader, device):
    net.eval()
    with torch.no_grad():
        # 遍历工况
        batch_idx = 0
        eval_loss = []
        for batch in eval_loader:
            input_data, y = batch
            split_num = len(input_data)
            loss_list = []
            for split_index in range(split_num):
                split_input = input_data[split_index][0].to(device)
                split_y = y[split_index][0].to(device)
                loss = custom_loss(net(split_input), split_y,Flag)
                loss_list.append(loss)

            eval_loss.append(sum(loss_list)/len(loss_list))
        net.train()
        return sum(eval_loss)/len(eval_loss)

def eval_net(net, eval_loader, result_pic_folder, device):
    net.eval()
    # net1 = MLPModel(11,2)
    # net1.load_state_dict(torch.load('./data/ckpt/cc_0919_v3_2048.pth', map_location=device))

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

                split_len = len(split_input)
                # if split_len < 50:
                #     continue
                '''
                air_temp_before_heat_exchange = split_input[:, 0]
                wind_vol = split_input[:, 1]
                step_2_refrigerant_temp = split_input[:, 2]
                step_1_refrigerant_temp = split_input[:, 3]
                # step_3_refrigerant_temp = split_input[:, 4]
                cab_heating_status_act_pos = split_input[:, 3]

                air_temp_after_heat_exchange = split_y[:, 0]
                refrigerant_mix_temp = split_y[:, 1]
                temp_p_h_5 = split_y[:, 2]

                pred_t_r_3, pred_t_w_3 = net(step_1_refrigerant_temp=step_1_refrigerant_temp,
                                             step_2_refrigerant_temp=step_2_refrigerant_temp,
                                             air_temp_before_heat_exchange=air_temp_before_heat_exchange,
                                             wind_vol=wind_vol, cab_heating_status_act_pos=cab_heating_status_act_pos)

                # refrigerant_mix_temp_mae = torch.mean(torch.abs(pred_t_r_1 - refrigerant_mix_temp))
                refrigerant_after_heat_exchange_mae = torch.mean(torch.abs(pred_t_r_3 - temp_p_h_5))
                air_temp_mae = torch.mean(torch.abs(pred_t_w_3 - air_temp_after_heat_exchange))

                # all_refrigerant_mix_temp_mae.append(refrigerant_mix_temp_mae.view(1, -1))
                all_refrigerant_after_heat_exchange_mae.append(refrigerant_after_heat_exchange_mae.view(1, -1))
                all_air_temp_mae.append(air_temp_mae.view(1, -1))
                '''
                # input_dim = input_data.shape[1]

                pred_output = net(split_input)
                # pred_output = net1(split_input)
                # zipped = zip(pred_output1, pred_output2)
                # sums = [x + y for x, y in zipped]
                # pred_output = tuple([s / 2 for s in sums])
                compressor_speed_real = split_y[:, 0]
                cab_heating_status_act_pos_real = split_y[:, 1]

                compressor_speed_pred = pred_output[0][:,0]
                cab_heating_status_act_pos_pred = pred_output[1][:,0]

                compressor_speed_mae = torch.mean(torch.abs(compressor_speed_pred - compressor_speed_real))
                compressor_speed_mae_prob = torch.abs(compressor_speed_pred - compressor_speed_real) / torch.clamp(compressor_speed_real,min=1)
                cab_heating_status_act_pos_mae = torch.mean(torch.abs(cab_heating_status_act_pos_pred - cab_heating_status_act_pos_real))
                cab_heating_status_act_pos_mae_prob = torch.abs(cab_heating_status_act_pos_pred - cab_heating_status_act_pos_real) / torch.clamp(cab_heating_status_act_pos_real,min=1)

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
                print('file_name', file_name, 'compressor_speed_mae', compressor_speed_mae,'cab_heating_status_act_pos_mae', cab_heating_status_act_pos_mae)
                pic_name = file_name + '.png'
                # print(series_list)
                draw_pred(series_list, series_name_list, result_pic_folder, pic_name)
                all_compressor_speed_mae.append(compressor_speed_mae.view(1, -1))
                all_cab_heating_status_act_pos_mae.append(cab_heating_status_act_pos_mae.view(1, -1))
                all_compressor_speed_prb_mae.extend(compressor_speed_mae_prob.view(1, -1))
                all_cab_heating_status_act_pos_prb_mae.extend(cab_heating_status_act_pos_mae_prob.view(1, -1))
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
    all_compressor_speed_prb_mae =torch.mean(all_compressor_speed_prb_mae)
    all_cab_heating_status_act_pos_prb_mae = torch.mean(all_cab_heating_status_act_pos_prb_mae)
    return mae_all_compressor_speed_mae, mae_all_cab_heating_status_act_pos_mae,all_compressor_speed_prb_mae,all_cab_heating_status_act_pos_prb_mae


if __name__ == '__main__':
    # A = torch.tensor([1.0,2,3,4,5,6,7])
    # B = torch.tensor([0.0,0,0,0,0,0,0])
    #
    # input = torch.cat((A.unsqueeze(1),A.unsqueeze(1)),dim=1)
    # target = torch.cat((B.unsqueeze(1),B.unsqueeze(1)),dim=1)
    # print(custom_loss(input,target,flag=1))
    #
    # print(custom_loss1(A,B,0,0))

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
    # pretrain_model_name = 'wind_temp_net.pth'
    # saved_model_name = 'wind_temp_net.pth'
    # pretrain_model_name = 'cmpr_control_net_v1.pth'
    # saved_model_name = 'cmpr_control_net_v2.pth'
    # pretrain_ckpt_path = os.path.join(ckpt_folder, pretrain_model_name)
    # pretrain_ckpt_path = './data/ckpt/cmpr_control_net_v1.pth'
    # saved_ckpt_path = os.path.join(ckpt_folder, saved_model_name)
    # saved_ckpt_path = './data/ckpt/cmpr_control_net_v1.pth'

    # 划分train eval
    seed = 4
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
    train_dataset_path = './data/cmpr_control_train_dataset_full.pkl'
    eval_dataset_path = './data/cmpr_control_eval_dataset_full.pkl'
    all_dataset_path = './data/cmpr_control_all_dataset_full.pkl'

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

        # all_dataset_list = []
        # features_list = []
        # label_list = []
        # for x_list in all_dataset.all_x:
        #     for x in x_list:
        #             features_list.extend(x.tolist())
        # for y_list in all_dataset.all_y:
        #     for y in y_list:
        #             label_list.extend(y.tolist())
        # features_list = torch.tensor(features_list)
        # label_list = torch.tensor(label_list)
        # all_dataset_list = torch.cat((features_list,label_list),dim=1)
        # all_dataset = all_dataset_list

        with open(train_dataset_path, 'wb') as f:
            pickle.dump(train_dataset, f)

        with open(eval_dataset_path, 'wb') as f:
            pickle.dump(eval_dataset, f)

        with open(all_dataset_path, 'wb') as f:
            pickle.dump(all_dataset, f)



    # 网络

    input_dim = 9
    output_dim = 2
    # net = MLPModel(input_dim=6, output_dim=1)
    net = MLPModel(input_size=6, output_dim=1, num_channels=[128]*2, kernel_size=5, dropout=0.5)
    # load_pretrain = False
    # model_params = {
    #     # 'input_size',C_in
    #     'input_size': 7,
    #     # 单步，预测未来一个时刻
    #     'output_size': 2,
    #     'num_channels': [128] * 2,
    #     'kernel_size': 5,
    #     'dropout': .0
    # }
    # net = tcn_net(**model_params)

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
