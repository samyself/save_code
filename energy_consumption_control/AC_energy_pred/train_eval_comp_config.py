import argparse
import os.path
import pickle

import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from tqdm import tqdm

import torch.nn as nn

import config_all as config
from data_utils import get_device
from dataset.dataset_comp_config import CompConfigDataset
from draw_pic import draw_pred
from model.model_comp_config import LinearRegressionModel


def train_net(net, train_epoch, train_loader, optimizer, device):
    # 收集所有批次的数据
    X_all = []
    y_all = []
    for X_batch, y_batch in train_loader:
        X_all.append(X_batch)
        y_all.append(y_batch)

    # 合并所有批次的数据
    X = torch.cat(X_all, dim=0)
    y = torch.cat(y_all, dim=0)

    # 添加偏置项
    X_b = torch.cat((torch.ones((X.shape[0], 1)), X), dim=1)

    # # 使用矩阵运算计算参数
    w = torch.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    # w = torch.pinverse(X_b) @ y
    print(w)

    # 更新模型中的权重
    net.linear.weight.data = w[1:].view_as(net.linear.weight.data)
    net.linear.bias.data = w[0]

    return 

def eval_net(net, eval_loader, result_pic_folder, device):
    net.eval()
    all_mae = []
    all_info = eval_loader.dataset.all_info
    with torch.no_grad():
        # 遍历工况
        batch_idx = 0
        for batch in tqdm(eval_loader):
            input_data, y = batch

            split_num = len(input_data)
            for split_index in range(split_num):
                split_input = input_data[split_index][0].to(device)
                split_y = y[split_index][0].to(device)
                pred = net(x=split_input)

                # pred = torch.exp(pred)
                # split_y = torch.exp(split_y)

                mae = torch.mean(torch.abs(pred - split_y))
                all_mae.append(mae.view(1, -1))

                # 可视化
                series_list = [
                    [pred, split_y],
                ]
                series_name_list = [
                    ['pred', 'ground_truth']
                ]
                file_name = all_info[batch_idx][split_index].replace('.csv', '').split('/')[-1].split('\\')[-1]
                pic_name = file_name + '.png'
                draw_pred(series_list, series_name_list, result_pic_folder, pic_name)

            batch_idx += 1

    net.train()

    # 计算指标
    all_mae = torch.cat(all_mae)
    mean_mae = torch.mean(all_mae)
    return mean_mae


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
    pretrain_model_name = 'comp_config_net_liner.pth'
    saved_model_name = 'comp_config_net_liner.pth'
    pretrain_ckpt_path = os.path.join(ckpt_folder, pretrain_model_name)
    saved_ckpt_path = os.path.join(ckpt_folder, saved_model_name)

    # 划分train eval
    seed = 111
    train_eval_rate = 0.6
    np.random.seed(seed)
    all_data_index = np.arange(len(all_data_path))
    np.random.shuffle(all_data_index)
    torch.manual_seed(seed)
    for path in all_data_path:
        print(path)

    train_final_place = int(train_eval_rate * len(all_data_index))
    train_index = all_data_index[:train_final_place]
    print(f'train_index:{train_index}')
    eval_index = all_data_index[train_final_place:]
    print(f'eval_index:{eval_index}')

    train_data_path = [all_data_path[i] for i in train_index]
    eval_data_path = [all_data_path[i] for i in eval_index]

    load_dataset = False
    train_dataset_path = './data/comp_config_train_dataset.pkl'
    eval_dataset_path = './data/comp_config_eval_dataset.pkl'
    all_dataset_path = './data/comp_config_all_dataset.pkl'
    if load_dataset:
        with open(train_dataset_path, 'rb') as f:
            train_dataset = pickle.load(f)

        with open(eval_dataset_path, 'rb') as f:
            eval_dataset = pickle.load(f)

        with open(all_dataset_path, 'rb') as f:
            all_dataset = pickle.load(f)
    else:
        train_dataset = CompConfigDataset(train_data_path, True)
        eval_dataset = CompConfigDataset(eval_data_path, False)
        all_dataset = CompConfigDataset(all_data_path, False)

        with open(train_dataset_path, 'wb') as f:
            pickle.dump(train_dataset, f)

        with open(eval_dataset_path, 'wb') as f:
            pickle.dump(eval_dataset, f)

        with open(all_dataset_path, 'wb') as f:
            pickle.dump(all_dataset, f)

    # 网络
    load_pretrain = False
    net = LinearRegressionModel(input_dim=6) 
    if load_pretrain:
        if os.path.exists(pretrain_ckpt_path):
            net.load_state_dict(torch.load(pretrain_ckpt_path, map_location=torch.device('cpu')))
    net.to(device)

    # 训练
    lr = config.lr
    train_batch_size = config.train_batch_size
    optimizer = optim.SGD(net.parameters(), lr=lr)
    train_epoch = config.train_epoch
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    train_net(net, train_epoch, train_loader, optimizer, device)
    torch.save(net.state_dict(), saved_ckpt_path)

    # 测试
    eval_batch_size = config.eval_batch_size
    cond_num = len(eval_dataset)

    eval_loader = DataLoader(all_dataset, batch_size=eval_batch_size, shuffle=False)
    mean_mae = eval_net(net, eval_loader, result_pic_folder, device)
    print('mean_mae', mean_mae)
