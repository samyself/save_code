import argparse
import os
#
# # 获取当前工作目录
# current_directory = os.getcwd()
# print(f"Current working directory: {current_directory}")
import pickle

from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from PycharmProjects.pythonProject2.dataset.dataset_soc_old import SOCBaseDataset
# from data_pre_filter import SOCBaseDataset

from common_func import draw_pred, get_loss_cur
# from model.model_cmpr_control_tcn import *
from PycharmProjects.pythonProject2.model.model_soc_lstm_old import MyModel
import torch.nn as nn
import torch
import pandas as pd

'''
压缩机控制模型
'''
def my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default='2', type=str, help='GPU ids')
    # 是否加载数据集
    parser.add_argument('--load_dataset', default=True, type=bool, help='')
    # 训练数据路径
    parser.add_argument('--train_dataset_path', default='../data/filter_data/pkl_data/seed111_1202_train_dataset_1000_filter.pkl', type=str, help='')
    # 评测数据路径
    parser.add_argument('--eval_dataset_path', default='../data/filter_data/pkl_data/seed111_1202_eval_dataset_1000_filter.pkl', type=str, help='')
    # 总体据路径
    parser.add_argument('--all_dataset_path', default='./data/energy_cunsum_pkl/high_press_pkl/high_press_all_dataset_checkpoint800_filter.pkl', type=str, help='')

    # 常修改参数
    # 学习率
    parser.add_argument('--lr', default=1e-2, type=float, help='')
    # 训练batch_size
    parser.add_argument('--train_batch_size', default=64, type=int, help='')
    # 权重衰减系数
    parser.add_argument('--weight_decay', default=0.01, type=float, help='')
    # 训练步长
    parser.add_argument('--train_epoch', default=40, type=int, help='')
    # 是否加载模型
    parser.add_argument('--load_pretrain', default=False, type=bool, help='')
    # 预训练模型路径
    parser.add_argument('--pretrain_ckpt_path', default='../data/ckpt/Soc_1202_lstm_v2.pth', type=str, help='')
    # 模型保存路径
    parser.add_argument('--saved_ckpt_path', default='../data/ckpt/Soc_1202_lstm_v2.pth', type=str, help='')
    # 是否训练
    parser.add_argument('--Flag_train', default=True, type=bool, help='')
    # 保存图片路径
    parser.add_argument('--result_pic_folder', default='../data/pic/Soc_1202_lstm_v2', type=str, help='')
    # 保存csv路径
    parser.add_argument('--result_csv_folder', default='../data/csv_res/Soc_1202_lstm_v2', type=str, help='')
    # 训练时自动保存的测试损失
    parser.add_argument('--best_loss', default= 200, type=int, help='')


    return parser

def clip_wights(net):
    for name, param in net.named_parameters():
        with torch.no_grad():
            # 保留四位小数
            param.data = torch.round(param.data * 10000) / 10000
            # 裁剪参数
            param.data = torch.clamp(param.data, -65535, 65535)

def XiangGuanDu(output, target):
    mean1 = torch.mean(output)
    mean2 = torch.mean(target)

    # 去均值
    tensor1_centered = output - mean1
    tensor2_centered = target - mean2

    # 计算协方差
    covariance = torch.dot(tensor1_centered, tensor2_centered) / (output.size(0) - 1)

    # 计算标准差
    std1 = torch.std(output, unbiased=True)
    std2 = torch.std(target, unbiased=True)

    # 计算皮尔逊相关系数
    pearson_correlation = covariance / (std1 * std2 + 1e-4)

    return pearson_correlation

"""
    output: batchsize*n(输出)
    target: batchsize*n(真实值)
    flag: 1-compressor_speed,2-cab_heating_status_act_po,3=1+2
    func: 损失函数
    return: 总loss
"""
def custom_loss(output, target):
    output = output.view(-1)
    target = target.view(-1)
    # mae_loss
    l2_loss = nn.MSELoss(reduction='mean')

    # mse_loss
    l1_loss = nn.L1Loss(reduction='mean')

    l3_loss = nn.SmoothL1Loss(reduction='mean')
    # Soc_r = XiangGuanDu(output, target)
    # Soc_r1 = np.corrcoef(output.detach().numpy(), target.detach().numpy())[0][1]

    # max_lim_loss = torch.mean(torch.relu(output - 110)) + 1

    # total_loss = l1_loss(output , target) + 3*torch.exp((1 - Soc_r)**1)
    total_loss = l3_loss(output, target)

    return total_loss


def train_net(net, train_epoch, train_loader, eval_loader, optimizer, device):
    criterion = custom_loss
    # criterion = nn.MSELoss(reduction='mean')
    train_loss = []
    eval_loss = []
    iterator = tqdm(range(train_epoch))

    A = best_loss
    for epoch_index in iterator:
        loss_list = []
        for batch_idx, (input_data, y) in enumerate(train_loader):
            input_data, y = input_data.to(device), y.to(device)
            pred_output = net(input_data)
            loss = criterion(pred_output, y)
            print(f'epoch {epoch_index},idx {batch_idx} ,loss{loss}')
            optimizer.zero_grad()
            loss.backward()
            # iterator.desc = f"epoch_index:{epoch_index},loss:{loss.item()}"
            # torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)
            optimizer.step()
            # clip_wights(net.Soc_model)
            loss_list.append(loss.item())
            # with torch.no_grad():
            #     net.cab_pos_n.clamp_(min=1.0, max=2.0)
        train_loss.append(sum(loss_list) / len(loss_list))
        eval_loss.append(eavl_loss(net, eval_loader, device))
        # eval_loss = [1000]
        print(
            f'Epoch [{epoch_index + 1}/{train_epoch}],Train Loss: {train_loss[-1]:.4f},eval Loss: {eval_loss[-1]:.4f}')

        if eval_loss[-1] < A:
            A = eval_loss[-1]
            torch.save(net.state_dict(), saved_ckpt_path)
            print('save model:eval loss is ',A)
    print('min loss=', A)
    return train_loss, eval_loss


def eavl_loss(net, eval_loader, device):
    net.eval()
    criterion = custom_loss

    with torch.no_grad():
        # 遍历工况
        batch_idx = 0
        eval_loss = []
        for batch in eval_loader:
            input_data, y = batch
            split_num = len(input_data)
            loss_list = []
            for split_index in range(split_num):
                split_input = input_data[0,:,:,:].to(device)
                split_y = y[split_index].to(device)

                # 多轮
                pred_output = net(split_input)
                # loss = torch.mean(torch.abs(pred_output - split_y))
                loss = criterion(pred_output, split_y)
                eval_loss.append(loss.item())

            # eval_loss.append(sum(loss_list) / len(loss_list))
        net.train()
        return sum(eval_loss) / len(eval_loss)


def eval_net(net, eval_loader, result_pic_folder, device):
    net.eval()
    criterion = custom_loss
    all_info = eval_loader.dataset.info
    all_Soc_mae = []
    all_Soc_prb_mae = []
    Soc_r_list = []

    # 记录所有数据
    all_data = []
    with torch.no_grad():
        # 遍历工况
        batch_idx = 0
        for batch in tqdm(eval_loader):
            input_data, y = batch
            split_num = len(input_data)

            for split_index in range(split_num):
                split_input = input_data[split_index].to(device)
                split_y = y[split_index]

                pred_output = net(split_input).cpu()
                Soc_real = split_y.view(-1)
                Bms_Soc = split_input[:, -1, 6].cpu()

                Soc_pred = pred_output.view(-1)

                Soc_mae = torch.mean(torch.abs(Soc_pred - Soc_real))
                Soc_erro_max = torch.max(torch.abs(Soc_pred - Soc_real))

                Soc_prob = torch.abs(Soc_pred - Soc_real) / torch.clamp(Soc_real,min=0)
                Soc_mae_prob = torch.mean(Soc_prob)
                Soc_r = np.corrcoef(Soc_pred,Soc_real)[0][1]

                if not np.isnan(Soc_r):
                    Soc_r_list.append(Soc_r)
                else:
                    Soc_r_list.append(0.0)

                # 可视化
                series_list = [
                    [Soc_pred, Soc_real, Bms_Soc],
                    [Soc_prob]
                ]
                series_name_list = [
                    # ['pred_refrigerant_mix_temp', 'refrigerant_mix_temp'],
                    ['Soc_pred', 'Soc_real', 'Bms_Soc'],
                    ['Soc_mae_prob']
                ]
                file_name = all_info[batch_idx][split_index].replace('.csv', '').split('/')[-1].split('\\')[-1]
                pic_name = file_name + '.png'
                # print(series_list)
                draw_pred(series_list, series_name_list, result_pic_folder, pic_name)
                # print(pic_name)
                print('file_name', file_name, 'Soc_mae', Soc_mae,'Soc_r', Soc_r,'Soc_mape',Soc_mae_prob, 'Soc_erro_max',Soc_erro_max)
                all_Soc_mae.append(Soc_mae)
                all_Soc_prb_mae.append(Soc_mae_prob)
                split_input = split_input.cpu()
                # 保存数据
                csv_dict = pd.DataFrame(
                    {'Volt': split_input[:,-1,0].numpy(),
                     'Curr': split_input[:,-1,1].numpy(),
                     'Ah': split_input[:,-1,2].numpy(),
                     'P_vxi': split_input[:,-1,3].numpy(),
                     'avg_current60': split_input[:,-1,4].numpy(),
                     'avg_voltage60': split_input[:,-1,5].numpy(),
                     'BmsSoc':split_input[:,-1,6].numpy(),
                     'Soc_pred':Soc_pred.numpy(),
                     'Soc_real':Soc_real.numpy()})
                result_csv_folder = args.result_csv_folder
                if not os.path.exists(result_csv_folder):
                    os.makedirs(result_csv_folder)
                result_csv_path = f'{result_csv_folder}/{file_name}.csv'
                csv_dict.to_csv(result_csv_path, index=False)

                # 9+2+2
                # all_data_temp = torch.cat((split_input,pred_output,split_y), dim=1)
                # all_data.append(all_data_temp.numpy())
            batch_idx += 1


    net.train()
    # np.save('AC_energy_pred/data/data_anyls/all_data1011.npy', all_data)

    # 计算指标
    # all_refrigerant_mix_temp_mae = torch.cat(all_refrigerant_mix_temp_mae)
    mae_all_Soc_mae = sum(all_Soc_mae)/len(all_Soc_mae)
    mae_all_Soc_prb_mae = torch.mean(torch.tensor(all_Soc_prb_mae))
    mae_Soc_r_list = sum(Soc_r_list)/len(Soc_r_list)


    print('mae_all_Soc_mae', mae_all_Soc_mae)
    print('mae_all_Soc_prb_mae', mae_all_Soc_prb_mae)
    print('mae_Soc_r_list', mae_Soc_r_list)
    return mae_all_Soc_mae, mae_all_Soc_prb_mae,mae_Soc_r_list


if __name__ == '__main__':

    parser = my_args()
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
    lr = args.lr
    train_batch_size = args.train_batch_size
    weight_decay = args.weight_decay
    train_epoch = args.train_epoch
    # 是否加载模型
    load_pretrain = args.load_pretrain
    pretrain_ckpt_path = args.pretrain_ckpt_path
    saved_ckpt_path = args.saved_ckpt_path
    # 是否训练
    Flag_train = args.Flag_train
    result_pic_folder = args.result_pic_folder
    # 训练时自动保存的测试损失
    best_loss = args.best_loss

    if not os.path.exists(result_pic_folder):
        os.makedirs(result_pic_folder)

    data_folder = '../data/filter_data/data_handle_BmsSoc_10_filter_v1'
    all_file_name = os.listdir(data_folder)
    all_data_path = [os.path.join(data_folder, file_name) for file_name in all_file_name]

    # 模型路径
    ckpt_folder = '../data/ckpt/'
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    # 划分train eval
    seed = 111
    train_eval_rate = 0.5
    np.random.seed(seed)
    all_data_index = np.arange(len(all_data_path))
    np.random.shuffle(all_data_index)
    torch.manual_seed(seed)

    train_final_place = int(train_eval_rate * len(all_data_index))
    choose_len = 1000
    train_index = all_data_index[train_final_place - choose_len:train_final_place]
    print('train_index len',len(train_index))
    eval_index = all_data_index[train_final_place:train_final_place + choose_len]
    print('eval_index len',len(eval_index))
    train_data_path = [all_data_path[i] for i in train_index]
    eval_data_path = [all_data_path[i] for i in eval_index]

    load_dataset = args.load_dataset
    train_dataset_path = args.train_dataset_path
    eval_dataset_path = args.eval_dataset_path
    # all_dataset_path = args.all_dataset_path

    if load_dataset:
        with open(train_dataset_path, 'rb') as f:
            train_dataset = pickle.load(f)

        with open(eval_dataset_path, 'rb') as f:
            eval_dataset = pickle.load(f)

        # with open(all_dataset_path, 'rb') as f:
        #     all_dataset = pickle.load(f)
    else:
        train_dataset = SOCBaseDataset(train_data_path, return_point=True)
        eval_dataset = SOCBaseDataset(eval_data_path, return_point=False)
        # all_dataset = CmprControlBaseDataset(all_data_path, False)

        with open(train_dataset_path, 'wb') as f:
            pickle.dump(train_dataset, f)

        with open(eval_dataset_path, 'wb') as f:
            pickle.dump(eval_dataset, f)

        # with open(all_dataset_path, 'wb') as f:
        #     pickle.dump(all_dataset, f)

    # 网络

    input_dim = 9
    output_dim = 2
    # net = MLPModel(input_dim=6, output_dim=1)
    net = MyModel()

    if load_pretrain:
        if os.path.exists(pretrain_ckpt_path):
            # net1 = torch.load(pretrain_ckpt_path, map_location=torch.device('cpu'))
            net.load_state_dict(torch.load(pretrain_ckpt_path, map_location=torch.device('cpu')))
    net.to(device).float()
    print('model in ',device)

    # torch.save(net.state_dict(), saved_ckpt_path)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    if Flag_train:
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)

        train_loss_list, eavl_loss_list = train_net(net, train_epoch, train_loader, eval_loader, optimizer, device)
        get_loss_cur(train_loss_list, eavl_loss_list,['train_loss', 'eval_loss'])

    # 测试
    mae_all_ac_pid_out_mae, mae_all_ac_pid_out_prb_mae,mae_ac_pid_out_r_list = eval_net(
        net, eval_loader, result_pic_folder, device)


