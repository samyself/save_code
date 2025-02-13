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
from model.model_actor import Actor
from dataset.dataset_actor import HeatingActorBaseDataset
from data_utils import get_device
from draw_pic import draw_pred


def train_net(net, train_epoch, train_loader, optimizer, device):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()

    iterator = tqdm(range(train_epoch))
    for epoch_index in iterator:
        for batch_idx, (input_data, y) in enumerate(train_loader):
            input_data, y = input_data.to(device), y.to(device)

            input_data = input_data.float()
            y = y.long()

            ags_pred, fan_pwm_pred = net(input_data)

            ags_openness_index = y[:, 0]
            cfan_pwm_index = y[:, 1]

            loss_ags_openness = criterion(ags_pred, ags_openness_index)
            loss_cfan_pwm = criterion(fan_pwm_pred, cfan_pwm_index)

            loss = loss_ags_openness + loss_cfan_pwm

            optimizer.zero_grad()
            loss.backward()
            iterator.desc = f"epoch_index:{epoch_index},loss:{loss.item()}"

            # 限制梯度大小
            scale = config.gard_scale
            for name, para in net.named_parameters():
                if para.grad is not None:
                    scaled_grad = scale * para.data * torch.sgn(para.grad)

                    # para.grad = scaled_grad
                    if para.shape != 0:
                        mask = torch.abs(scaled_grad) < torch.abs(para.grad)
                        para.grad[mask] = scaled_grad[mask]
                    else:
                        if torch.abs(scaled_grad) < torch.abs(para.grad):
                            para.grad = scaled_grad

            optimizer.step()

    return


def eval_net(net, eval_loader, result_pic_folder, device):
    net.eval()

    all_ags_index_mae = []
    all_fan_pwm_index_mae = []

    all_info = eval_loader.dataset.all_info
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

                split_input = split_input.float()
                split_y = split_y.long()

                ags_openness_index = split_y[:, 0]
                cfan_pwm_index = split_y[:, 1]

                # split_len = len(split_input)

                ags_pred, fan_pwm_pred = net(split_input)

                ags_choice = torch.argmax(ags_pred, dim=-1)
                fan_pwm_choice = torch.argmax(fan_pwm_pred, dim=-1)

                ags_index_mae = torch.mean(torch.abs(ags_choice - ags_openness_index).float())
                fan_pwm_index_mae = torch.mean(torch.abs(fan_pwm_choice - cfan_pwm_index).float())

                # all_refrigerant_mix_temp_mae.append(refrigerant_mix_temp_mae.view(1, -1))
                all_ags_index_mae.append(ags_index_mae.view(1, -1))
                all_fan_pwm_index_mae.append(fan_pwm_index_mae.view(1, -1))

                # 可视化
                series_list = [

                    [ags_choice, ags_openness_index],
                    [fan_pwm_choice, cfan_pwm_index],
                ]
                series_name_list = [

                    ['ags_choice', 'ags_openness_index'],
                    ['fan_pwm_choice', 'cfan_pwm_index'],
                ]
                file_name = all_info[batch_idx][split_index].replace('.csv', '').split('/')[-1].split('\\')[-1]
                pic_name = file_name + '.png'
                draw_pred(series_list, series_name_list, result_pic_folder, pic_name)

            batch_idx += 1

    net.train()

    # 计算指标
    all_ags_index_mae = torch.cat(all_ags_index_mae)
    all_fan_pwm_index_mae = torch.cat(all_fan_pwm_index_mae)

    # mean_refrigerant_mix_temp_mae = torch.mean(all_refrigerant_mix_temp_mae)
    mean_ags_index_mae = torch.mean(all_ags_index_mae)
    mean_fan_pwm_index_mae = torch.mean(all_fan_pwm_index_mae)
    return mean_ags_index_mae, mean_fan_pwm_index_mae


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default='2', type=str, help='GPU ids')
    args = parser.parse_args()
    device = get_device(device_id=args.device_id)

    result_pic_folder = './data/train_result/actor/pic'
    if not os.path.exists(result_pic_folder):
        os.makedirs(result_pic_folder)

    data_folder = './data/csv'
    all_file_name = os.listdir(data_folder)
    all_data_path = [os.path.join(data_folder, file_name) for file_name in all_file_name]

    # 模型路径
    ckpt_folder = './data/ckpt/'
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    pretrain_model_name = 'actor.pth'
    saved_model_name = 'actor.pth'
    pretrain_ckpt_path = os.path.join(ckpt_folder, pretrain_model_name)
    saved_ckpt_path = os.path.join(ckpt_folder, saved_model_name)

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
    train_dataset_path = './data/actor_train_dataset.pkl'
    eval_dataset_path = './data/actor_eval_dataset.pkl'
    all_dataset_path = './data/actor_all_dataset.pkl'
    if load_dataset:
        with open(train_dataset_path, 'rb') as f:
            train_dataset = pickle.load(f)

        with open(eval_dataset_path, 'rb') as f:
            eval_dataset = pickle.load(f)

        with open(all_dataset_path, 'rb') as f:
            all_dataset = pickle.load(f)
    else:
        train_dataset = HeatingActorBaseDataset(train_data_path, True)
        eval_dataset = HeatingActorBaseDataset(eval_data_path, False)
        all_dataset = HeatingActorBaseDataset(all_data_path, False)

        with open(train_dataset_path, 'wb') as f:
            pickle.dump(train_dataset, f)

        with open(eval_dataset_path, 'wb') as f:
            pickle.dump(eval_dataset, f)

        with open(all_dataset_path, 'wb') as f:
            pickle.dump(all_dataset, f)

    # 网络
    load_pretrain = False
    ags_dim = 16
    fan_pwm_dim = 101
    input_dim = 9
    net = Actor(input_dim=input_dim, out_dim=[ags_dim, fan_pwm_dim])
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

    # eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
    eval_loader = DataLoader(all_dataset, batch_size=eval_batch_size, shuffle=False)
    mean_ags_index_mae, mean_fan_pwm_index_mae = eval_net(net, eval_loader, result_pic_folder, device)

    # print('mean_refrigerant_mix_temp_mae', mean_refrigerant_mix_temp_mae)
    print('mean_ags_index_mae', mean_ags_index_mae)
    print('mean_fan_pwm_index_mae', mean_fan_pwm_index_mae)
