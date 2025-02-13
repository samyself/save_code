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
from AC_energy_pred.model.my_common import get_loss_cur

# from data_utils import get_device
from dataset.dataset_acpid_out import AcPidOutBaseDataset
from draw_pic import draw_pred
# from model.model_cmpr_control_tcn import *
from model.model_ac_pid_out_PID_OK import *
import pandas as pd
from model.model_ac_kprate_lemans import MyModel as ACKPRate_model

'''
压缩机控制模型
'''
def my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default='2', type=str, help='GPU ids')
    # 是否加载数据集
    parser.add_argument('--load_dataset', default=False, type=bool, help='')
    # 训练数据路径
    parser.add_argument('--train_dataset_path', default='./data/energy_cunsum_data/pkl_file/acc_pid_out/seed2_acpidout_train_20250208_test.pkl', type=str, help='')
    # 评测数据路径
    parser.add_argument('--eval_dataset_path', default='./data/energy_cunsum_data/pkl_file/acc_pid_out/seed2_acpidout_eval_20250208_test.pkl', type=str, help='')
    # 总体据路径
    parser.add_argument('--all_dataset_path', default='./data/energy_cunsum_data/pkl_file/acc_pid_out/seed2_acpidout_all_10250123_test.pkl', type=str, help='')

    # 常修改参数
    # 学习率
    parser.add_argument('--lr', default=1e-2, type=float, help='')
    # 训练batch_size
    parser.add_argument('--train_batch_size', default=1, type=int, help='')
    # 权重衰减系数
    parser.add_argument('--weight_decay', default=0.01, type=float, help='')
    # 训练步长
    parser.add_argument('--train_epoch', default=20, type=int, help='')
    # 是否加载模型
    parser.add_argument('--load_pretrain', default=False, type=bool, help='')
    # 预训练模型路径
    parser.add_argument('--pretrain_ckpt_path', default='./data/ckpt/ac_20250208_PID.pth', type=str, help='')
    # 模型保存路径
    parser.add_argument('--saved_ckpt_path', default='data/ckpt/ac_20250208_PID.pth', type=str, help='')
    # 允许更改的参数1：转速，2：开度，3：都更改
    parser.add_argument('--Flag', default=1, type=int, help='')
    # 是否训练
    parser.add_argument('--Flag_train', default=False, type=bool, help='')
    # 保存图片路径
    parser.add_argument('--result_pic_folder', default='./data/train_result/pic/ac_20250123_PID_lemans_test1', type=str, help='')
    # 训练时自动保存的测试损失
    parser.add_argument('--best_loss', default=400, type=int, help='')

    return parser
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
    l1_loss = nn.L1Loss(reduction='mean')
    # mse_loss
    # l2_loss = nn.MSELoss(reduction='mean')

    total_loss = l1_loss(output , target)

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
            split_num = len(input_data)
            for split_index in range(split_num):
                split_input = input_data[split_index][0].to(device)
                split_y = y[split_index][0].to(device)
                split_len = len(split_input)
                pred_output = torch.zeros_like(split_y)
                loss = 0
                for i in range(split_len):
                    if i % 30 == 0:
                        net.last_ac_pid_out_hp = split_input[i, 0].unsqueeze(0)

                    pred_output[i] = net(split_input[i])
                    # if i % 30 == 0 and i>0:
                    #     loss = loss + criterion(pred_output[i-1], split_y[i-1])
                torch.autograd.set_detect_anomaly(True)
                loss = criterion(pred_output, split_y)
                optimizer.zero_grad()
                loss.backward()
                loss_list.append(loss.item())
                optimizer.step()

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
                split_input = input_data[split_index][0].to(device)
                split_y = y[split_index][0].to(device)

                split_len = len(split_input)
                pred_output = torch.zeros_like(split_y)

                for i in range(split_len):
                    if i % 30 == 0:
                        net.last_ac_pid_out_hp = split_input[i,0].unsqueeze(0)
                    pred_output[i] = net(split_input[i])


                #     # net.last_ac_pid_out_hp = pred_output[i]

                # 多轮
                # pred_output = net(split_input)
                loss = criterion(pred_output, split_y)
                loss_list.append(loss)

            eval_loss.append(sum(loss_list) / len(loss_list))
        net.train()
        return sum(eval_loss) / len(eval_loss)


def eval_net(net, eval_loader, result_pic_folder, device):
    net.eval()
    all_info = eval_loader.dataset.all_info
    all_ac_pid_out_mae = []
    all_ac_pid_out_prb_mae = []
    ac_pid_out_r_list = []

    # 记录所有数据
    all_data = []
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
                pred_output = torch.zeros_like(split_y)
                all_res = torch.zeros((split_len, 13))
                net.num_LP = 0
                for i in range(split_len):
                    if i % 30 == 0:
                        # net.last_ac_pid_out_hp = split_input[i,0].unsqueeze(0)
                        pass
                    else:
                        split_input[i,0] = pred_output[i-1]
                    # if (i-1) % 30 == 0:
                    #     net.Ivalue = split_input[i-1,15].unsqueeze(0)

                    # all_res[i,:] = net(split_input[i])
                    pred_output[i] = net(split_input[i])[0,0]
                    # net.last_ac_pid_out_hp = pred_output[i]
                print(f'总长度为：{split_len}  ,用到了低压的长度为：{net.num_LP}')
                # pred_output = net(split_input)
                ac_pid_out_real = split_y[:, 0]

                ac_pid_out_pred = pred_output[:,0]

                # 保存数据
                csv_dict = pd.DataFrame(
                    {'High_press': split_input[:,5].numpy(),
                     'Low_press': split_input[:,4].numpy(),
                     'Aim_high_press': split_input[:,7].numpy(),
                     'Aim_low_press': split_input[:,6].numpy(),
                     'temp_p_h_2' : split_input[:,12].numpy(),
                     'temp_amb' : split_input[:,1].numpy(),
                     'real_ac_pid_out': split_y[:,0].numpy(),
                     'Kp_HP': all_res[:,1].numpy(),
                     'Ki_HP': all_res[:,2].numpy(),
                     'Kd_HP': all_res[:,3].numpy(),
                     'Iout_HP': all_res[:,4].numpy(),
                     'Diffout_HP': all_res[:,5].numpy(),
                     'HP_AD': all_res[:,6].numpy(),
                     'Kp_LP': all_res[:, 7].numpy(),
                     'Ki_LP': all_res[:, 8].numpy(),
                     'Kd_LP': all_res[:, 9].numpy(),
                     'Iout_LP': all_res[:, 10].numpy(),
                     'Diffout_LP': all_res[:, 11].numpy(),
                     'LP_AD': all_res[:, 12].numpy(),
                     'pred_ac_pid_out': pred_output[:,0].numpy(),
                     })
                file_name = all_info[batch_idx][split_index].replace('.csv', '').split('/')[-1].split('\\')[-1]
                result_csv_folder = './data/train_result/csv/ac_20250205_PID_Pred_lemans5_1'
                if not os.path.exists(result_csv_folder):
                    os.makedirs(result_csv_folder)
                result_csv_path = f'{result_csv_folder}/{file_name}.csv'
                csv_dict.to_csv(result_csv_path, index=False)

                ac_pid_out_mae = torch.mean(torch.abs(ac_pid_out_pred - ac_pid_out_real))
                ac_pid_out_prob = torch.abs(ac_pid_out_pred - ac_pid_out_real) / torch.clamp(ac_pid_out_real,min=0)
                ac_pid_out_mae_prob = torch.mean(ac_pid_out_prob)
                ac_pid_out_r = np.corrcoef(ac_pid_out_pred,ac_pid_out_real)[0][1]

                if np.isnan(ac_pid_out_r):
                    data_gap = ac_pid_out_pred - ac_pid_out_real
                    if (data_gap[0] == data_gap).all():
                        ac_pid_out_r = 1
                    else:
                        ac_pid_out_r = 0


                if not np.isnan(ac_pid_out_r):
                    ac_pid_out_r_list.append(ac_pid_out_r)

                # 查看高压与pid的走向
                aim_hi_pressure = split_input[:, 7]
                hi_pressure = split_input[:, 5]
                diff_hi_press = (aim_hi_pressure - hi_pressure)*0.0244
                lo_pressure = split_input[:, 4]
                aim_lo_pressure = split_input[:, 6]
                diff_low_press = (lo_pressure - aim_lo_pressure)*0.0996

                # 可视化
                series_list = [
                    [ac_pid_out_pred, ac_pid_out_real],
                    [ac_pid_out_prob],
                    [diff_hi_press, diff_low_press]
                ]
                series_name_list = [
                    # ['pred_refrigerant_mix_temp', 'refrigerant_mix_temp'],
                    ['ac_pid_out_pred', 'ac_pid_out_real'],
                    ['ac_pid_out_mae_prob'],
                    ['diff_hi_press', 'diff_low_press']
                ]
                file_name = all_info[batch_idx][split_index].replace('.csv', '').split('/')[-1].split('\\')[-1]
                pic_name = file_name + '.png'
                # print(series_list)
                draw_pred(series_list, series_name_list, result_pic_folder, pic_name)
                # print(pic_name)
                print('file_name', file_name, 'ac_pid_out_mae', ac_pid_out_mae,'ac_pid_out_r', ac_pid_out_r,'ac_pid_out_mape',ac_pid_out_mae_prob)
                all_ac_pid_out_mae.append(ac_pid_out_mae.view(1, -1))
                all_ac_pid_out_prb_mae.append(ac_pid_out_mae_prob)

                # 9+2+2
                # all_data_temp = torch.cat((split_input,pred_output,split_y), dim=1)
                # all_data.append(all_data_temp.numpy())
            batch_idx += 1


    net.train()
    # np.save('AC_energy_pred/data/data_anyls/all_data1011.npy', all_data)

    # 计算指标
    # all_refrigerant_mix_temp_mae = torch.cat(all_refrigerant_mix_temp_mae)
    mae_all_ac_pid_out_mae = sum(all_ac_pid_out_mae)/len(all_ac_pid_out_mae)
    mae_all_ac_pid_out_prb_mae = torch.mean(torch.tensor(all_ac_pid_out_prb_mae))
    mae_ac_pid_out_r_list = sum(ac_pid_out_r_list)/len(ac_pid_out_r_list)


    print('mae_all_ac_pid_out_mae', mae_all_ac_pid_out_mae)
    print('mae_all_ac_pid_out_prb_mae', mae_all_ac_pid_out_prb_mae)
    print('mae_ac_pid_out_r_list', mae_ac_pid_out_r_list)
    return mae_all_ac_pid_out_mae, mae_all_ac_pid_out_prb_mae,mae_ac_pid_out_r_list


if __name__ == '__main__':
    car_type = 'lemans_adv'
    # lemans_low lemans_mid lemans_adv
    parser = my_args()
    args = parser.parse_args()
    device = 'cpu'
    lr = args.lr
    train_batch_size = args.train_batch_size
    weight_decay = args.weight_decay
    train_epoch = args.train_epoch
    # 是否加载模型
    load_pretrain = args.load_pretrain
    pretrain_ckpt_path = args.pretrain_ckpt_path
    saved_ckpt_path = args.saved_ckpt_path
    # 允许更改的参数1：转速，2：开度，3：都更改
    Flag = args.Flag
    # 是否训练
    Flag_train = args.Flag_train
    result_pic_folder = args.result_pic_folder
    # 训练时自动保存的测试损失
    best_loss = args.best_loss

    if not os.path.exists(result_pic_folder):
        os.makedirs(result_pic_folder)

    data_folder = 'D:/Code/code/submit/eng_test/energy_consumption_control/AC_energy_pred/data/pre_filter/csv_lemans_all_zfq/lemans_5'
    # data_folder = 'D:/Code/code/submit/eng_test/energy_consumption_control/AC_energy_pred/data/pre_filter/badcase_csv'

    all_file_name = os.listdir(data_folder)
    # all_file_name = ['2024-11-13 21_59_29_MX11_lemans-5, Lemans-THEM_A_E4U0_T161_LV025-20℃城市工况auto24三人夜间舒适性3.csv']
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
    seed = 2
    train_eval_rate = 0.6
    np.random.seed(seed)
    all_data_index = np.arange(len(all_data_path))
    np.random.shuffle(all_data_index)
    torch.manual_seed(seed)

    train_final_place = int(train_eval_rate * len(all_data_index))
    train_index = all_data_index[:3]
    eval_index = all_data_index[:]

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
        train_dataset = AcPidOutBaseDataset(train_data_path, False,car_type=car_type)
        eval_dataset = AcPidOutBaseDataset(eval_data_path, False,car_type=car_type)
        # all_dataset = CmprControlBaseDataset(all_data_path, False)

        with open(train_dataset_path, 'wb') as f:
            pickle.dump(train_dataset, f)

        with open(eval_dataset_path, 'wb') as f:
            pickle.dump(eval_dataset, f)

        # with open(all_dataset_path, 'wb') as f:
        #     pickle.dump(all_dataset, f)

    # 网络

    # net = MLPModel(input_dim=6, output_dim=1)
    net = MyModel(car_type=car_type)

    if load_pretrain:
        if os.path.exists(pretrain_ckpt_path):
            # net1 = torch.load(pretrain_ckpt_path, map_location=torch.device('cpu'))
            net.load_state_dict(torch.load(pretrain_ckpt_path, map_location=torch.device('cpu')))
    net.to(device)
    # ak_model = ACKPRate_model(car_type=car_type)
    #
    # ak_model.load_state_dict(torch.load('./data/ckpt/ak_20250208_v1.pth', map_location=torch.device('cpu')))
    #
    # net.MLP1.load_state_dict(ak_model.MLP1.state_dict())

    # torch.save(net.state_dict(), saved_ckpt_path)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False)

    if Flag_train:
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)

        train_loss_list, eavl_loss_list = train_net(net, train_epoch, train_loader, eval_loader, optimizer, device)
        get_loss_cur(train_loss_list, eavl_loss_list,['train_loss', 'eval_loss'])

    # 测试
    eval_batch_size = config.eval_batch_size
    cond_num = len(eval_dataset)

    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
    mae_all_ac_pid_out_mae, mae_all_ac_pid_out_prb_mae,mae_ac_pid_out_r_list = eval_net(
        net, eval_loader, result_pic_folder, device)


