import argparse
import os.path
import pickle

import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from tqdm import tqdm

import torch.nn as nn

import config
from data_utils import get_device
from dataset.dataset_cmpr_control import CmprControlBaseDataset
from draw_pic import draw_pred
from model.model_cmpr_control import *
from model.my_common import get_loss_cur
'''
压缩机控制模型
'''
def my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default='2', type=str, help='GPU ids')
    # 是否加载数据集
    parser.add_argument('--load_dataset', default=True, type=bool, help='')
    # 训练数据路径
    parser.add_argument('--train_dataset_path', default='./data/energy_cunsum_data/pkl_file/acc_pid_out/seed2_comsped_train_normal1016.pkl', type=str, help='')
    # 评测数据路径
    parser.add_argument('--eval_dataset_path', default='./data/energy_cunsum_data/pkl_file/acc_pid_out/seed2_comsped_eval_normal1016.pkl', type=str, help='')
    # 总体据路径
    parser.add_argument('--all_dataset_path', default='./data/energy_cunsum_pkl/cmpr_control_all_dataset_full_seed2_up800.pkl', type=str, help='')

    # 常修改参数
    # 学习率
    parser.add_argument('--lr', default=1e-3, type=float, help='')
    # 训练batch_size
    parser.add_argument('--train_batch_size', default=1024, type=int, help='')
    # 权重衰减系数
    parser.add_argument('--weight_decay', default=0.01, type=float, help='')
    # 训练步长
    parser.add_argument('--train_epoch', default=40, type=int, help='')
    # 是否加载模型
    parser.add_argument('--load_pretrain', default=False, type=bool, help='')
    # 预训练模型路径
    parser.add_argument('--pretrain_ckpt_path', default='./data/ckpt/cs_1105_v2.pth', type=str, help='')
    # 模型保存路径
    parser.add_argument('--saved_ckpt_path', default='data/ckpt/cs_1105_v2.pth', type=str, help='')
    # 允许更改的参数1：转速，2：开度，3：都更改
    parser.add_argument('--Flag', default=1, type=int, help='')
    # 是否训练
    parser.add_argument('--Flag_train', default=True, type=bool, help='')
    # 保存图片路径
    parser.add_argument('--result_pic_folder', default='./data/train_result/pic/cs_1105_v2', type=str, help='')
    # 训练时自动保存的测试损失
    parser.add_argument('--best_loss', default=408, type=int, help='')

    return parser




def train_net(net, train_epoch, train_loader,eval_loader, optimizer, device):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss(reduction='mean')
    train_loss = []
    eval_loss = []
    iterator = tqdm(range(train_epoch))
    # A = 7
    for epoch_index in iterator:
        loss_list = []
        for batch_idx, (input_data, y) in enumerate(train_loader):
            input_data, y = input_data.to(device), y.to(device)

            pred_output = net(input_data)
            compressor_speed_real = y[:,0]
            cab_heating_status_act_pos_real = y[:,1]

            compressor_speed_pred = pred_output[0][:,0]
            cab_heating_status_act_pos_pred = pred_output[1][:,0]

            # loss_compressor_speed =  criterion(compressor_speed_pred, compressor_speed_real)
            # loss_act_pos =  criterion(cab_heating_status_act_pos_pred, cab_heating_status_act_pos_real)
            loss_compressor_speed =  torch.sqrt(criterion(compressor_speed_pred, compressor_speed_real))
            loss_act_pos =  torch.sqrt(criterion(cab_heating_status_act_pos_pred, cab_heating_status_act_pos_real))

            loss = loss_act_pos + loss_compressor_speed
            # loss = loss_compressor_speed
            optimizer.zero_grad()
            loss.backward()

            # iterator.desc = f"epoch_index:{epoch_index},loss:{loss.item()}"
            loss_list.append(loss.item())
            # 限制梯度大小
            '''
            scale = config.gard_scale
            for name, para in net.named_parameters():
                if torch.any(para.grad !=0):
                    scaled_grad = scale * para.data * torch.sgn(para.grad)
                    # 创建一个布尔tensor，当scaled_grad的绝对值小于para.grad的绝对值时为True
                    condition = torch.abs(scaled_grad) < torch.abs(para.grad)
                    # 使用这个布尔tensor来选择性地更新梯度
                    para.grad = torch.where(condition, scaled_grad, para.grad)
            '''
            optimizer.step()

        train_loss.append(sum(loss_list)/len(loss_list))
        print('train_loss', train_loss[-1])
        eval_loss.append(eavl_loss(net,eval_loader,device))
        print('eval_loss',eval_loss[-1])

        # if train_loss[-1]<A and eval_loss[-1]<A:
        #     torch.save(net.state_dict(), saved_ckpt_path)

        # if train_loss[-1]<eval_loss[-1]*0.8:
        #     break
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

                pred_output = net(split_input)

                compressor_speed_real = split_y[:, 0]
                cab_heating_status_act_pos_real = split_y[:, 1]

                compressor_speed_pred = pred_output[0][:,0]
                cab_heating_status_act_pos_pred = pred_output[1][:,0]

                compressor_speed_mae = torch.mean(torch.abs(compressor_speed_pred - compressor_speed_real))
                cab_heating_status_act_pos_mae = torch.mean(torch.abs(cab_heating_status_act_pos_pred - cab_heating_status_act_pos_real))
                loss = compressor_speed_mae + cab_heating_status_act_pos_mae
                loss_list.append(loss)

            eval_loss.append(sum(loss_list)/len(loss_list))
        net.train()
        return sum(eval_loss)/len(eval_loss)

def eval_net(net, eval_loader, result_pic_folder, device):
    net.eval()
    # all_refrigerant_mix_temp_mae = []
    # all_refrigerant_after_heat_exchange_mae = []
    # all_air_temp_mae = []
    all_info = eval_loader.dataset.all_info
    all_compressor_speed_mae = []
    all_cab_heating_status_act_pos_mae = []
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

                compressor_speed_real = split_y[:, 0]
                cab_heating_status_act_pos_real = split_y[:, 1]

                compressor_speed_pred = pred_output[0][:,0]
                cab_heating_status_act_pos_pred = pred_output[1][:,0]

                compressor_speed_mae = torch.mean(torch.abs(compressor_speed_pred - compressor_speed_real))
                cab_heating_status_act_pos_mae = torch.mean(torch.abs(cab_heating_status_act_pos_pred - cab_heating_status_act_pos_real))
                # 可视化
                series_list = [
                    # [pred_t_r_1, refrigerant_mix_temp],
                    [compressor_speed_pred, compressor_speed_real],
                    [cab_heating_status_act_pos_pred, cab_heating_status_act_pos_real]
                ]
                series_name_list = [
                    # ['pred_refrigerant_mix_temp', 'refrigerant_mix_temp'],
                    ['pred_compressor_speed', 'compressor_speed'],
                    ['pred_cab_heating_status_act_pos', 'cab_heating_status_act_pos']
                ]
                file_name = all_info[batch_idx][split_index].replace('.csv', '').split('/')[-1].split('\\')[-1]
                pic_name = file_name + '.png'
                # print(series_list)
                draw_pred(series_list, series_name_list, result_pic_folder, pic_name)
                all_compressor_speed_mae.append(compressor_speed_mae.view(1, -1))
                all_cab_heating_status_act_pos_mae.append(cab_heating_status_act_pos_mae.view(1, -1))

            batch_idx += 1

    net.train()

    # 计算指标
    # all_refrigerant_mix_temp_mae = torch.cat(all_refrigerant_mix_temp_mae)
    all_compressor_speed_mae = torch.cat(all_compressor_speed_mae)
    all_cab_heating_status_act_pos_mae = torch.cat(all_cab_heating_status_act_pos_mae)

    # mean_refrigerant_mix_temp_mae = torch.mean(all_refrigerant_mix_temp_mae)
    mae_all_compressor_speed_mae = torch.mean(all_compressor_speed_mae)
    mae_all_cab_heating_status_act_pos_mae = torch.mean(all_cab_heating_status_act_pos_mae)
    return mae_all_compressor_speed_mae, mae_all_cab_heating_status_act_pos_mae


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default='2', type=str, help='GPU ids')
    args = parser.parse_args()
    device = get_device(device_id=args.device_id)

    result_pic_folder = './data/train_result/pic'
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
    pretrain_model_name = 'cc_0918_v2.pth'
    saved_model_name = 'cc_0918_v2.pth'
    pretrain_ckpt_path = os.path.join(ckpt_folder, pretrain_model_name)
    # pretrain_ckpt_path = './data/ckpt/cmpr_control_net_v1.pth'
    saved_ckpt_path = os.path.join(ckpt_folder, saved_model_name)
    # saved_ckpt_path = './data/ckpt/cmpr_control_net_v1.pth'

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

    load_dataset = True
    train_dataset_path = './data/cmpr_control_train_dataset.pkl'
    eval_dataset_path = './data/cmpr_control_eval_dataset.pkl'
    all_dataset_path = './data/cmpr_control_all_dataset.pkl'

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
    load_pretrain = True
    input_dim = 9
    output_dim = 2
    net = MLPModel(input_dim=input_dim, output_dim=output_dim)
    if load_pretrain:
        if os.path.exists(pretrain_ckpt_path):
            net.load_state_dict(torch.load(pretrain_ckpt_path, map_location=torch.device('cpu')))
    net.to(device)

    # 训练
    lr = config.lr
    train_batch_size = config.train_batch_size
    weight_decay = 0.01
    # optimizer = optim.SGD(net.parameters(), lr=lr)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    train_epoch = config.train_epoch
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False)
    train_loss_list,eavl_loss_list = train_net(net, train_epoch, train_loader,eval_loader, optimizer, device)

    get_loss_cur(train_loss_list,eavl_loss_list,['train loss','eval loss'])
    torch.save(net.state_dict(), saved_ckpt_path)

    # 测试
    eval_batch_size = config.eval_batch_size
    cond_num = len(eval_dataset)


    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
    mae_all_compressor_speed_mae, mae_all_cab_heating_status_act_pos_mae = eval_net(net, eval_loader, result_pic_folder, device)

    # print('mean_refrigerant_mix_temp_mae', mean_refrigerant_mix_temp_mae)
    print('mae_all_compressor_speed_mae', mae_all_compressor_speed_mae)
    print('mae_all_cab_heating_status_act_pos_mae', mae_all_cab_heating_status_act_pos_mae)
