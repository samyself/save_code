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
from data_utils import get_device
from dataset.dataset_cmpr_control import CmprControlBaseDataset
from draw_pic import draw_pred
# from model.model_cmpr_control_tcn import *
from model.model_cmpr_control_lemans import *
# from torch.cuda.amp import GradScaler, autocast

'''
压缩机控制模型
'''
def my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default='2', type=str, help='GPU ids')
    # 是否加载数据集
    parser.add_argument('--load_dataset', default=False, type=bool, help='')
    # 训练数据路径
    parser.add_argument('--train_dataset_path', default='./data/energy_cunsum_pkl/compresorr_speed_pkl/seed2_cc_trian_1011_filter.pkl', type=str, help='')
    # 评测数据路径
    parser.add_argument('--eval_dataset_path', default='./data/energy_cunsum_pkl/compresorr_speed_pkl/seed2_cc_eval_1011_filter.pkl', type=str, help='')
    # 总体据路径
    parser.add_argument('--all_dataset_path', default='./data/energy_cunsum_pkl/cmpr_control_all_dataset_full_seed2_up800.pkl', type=str, help='')

    # 常修改参数
    # 学习率
    parser.add_argument('--lr', default=1e-4, type=float, help='')
    # 训练batch_size
    parser.add_argument('--train_batch_size', default=256, type=int, help='')
    # 权重衰减系数
    parser.add_argument('--weight_decay', default=0.01, type=float, help='')
    # 训练步长
    parser.add_argument('--train_epoch', default=40, type=int, help='')
    # 是否加载模型
    parser.add_argument('--load_pretrain', default=True, type=bool, help='')
    # 预训练模型路径
    parser.add_argument('--pretrain_ckpt_path', default='./data/ckpt/cc_1225_v3.pth', type=str, help='')
    # 模型保存路径
    parser.add_argument('--saved_ckpt_path', default='data/ckpt/cc_1225_v3.pth', type=str, help='')
    # 允许更改的参数1：转速，2：开度，3：都更改
    parser.add_argument('--Flag', default=2, type=int, help='')
    # 是否训练
    parser.add_argument('--Flag_train', default=False, type=bool, help='')
    # 保存图片路径
    parser.add_argument('--result_pic_folder', default='./data/train_result/pic/cc_1225_v2_test3', type=str, help='')
    # 训练时自动保存的测试损失
    parser.add_argument('--best_loss', default=2, type=int, help='')

    return parser


def clip_wights(net):
    for name, param in net.named_parameters():
        with torch.no_grad():
            # 保留四位小数
            # param.data = torch.clamp(param.data*1e3, min=0)
            # param.data = param.data/1e3
            # 裁剪参数
            param.data = torch.clamp(param.data, -65504, 65504)

"""
    output: batchsize*n(输出)
    target: batchsize*n(真实值)
    flag: 1-compressor_speed,2-cab_heating_status_act_po,3=1+2
    func: 损失函数
    return: 总loss
"""
def custom_loss(output, target, flag):
    Sloss = nn.L1Loss(reduction='mean')
    # Sloss = nn.SmoothL1Loss(beta=1000)
    if flag == 1 or flag == 2:
        total_loss = Sloss(output[:, flag - 1], target[:, flag - 1])
    else:
        total_loss = torch.tensor(0.0)
        for i in range(flag - 1):
            loss_temp = Sloss(output[:, i], target[:, i])
            total_loss = total_loss + loss_temp

    return total_loss


def train_net(net, train_epoch, train_loader, eval_loader, optimizer, device):
    criterion = custom_loss
    # criterion = nn.MSELoss(reduction='mean')
    train_loss = []
    eval_loss = []
    iterator = tqdm(range(train_epoch))
    # 创建梯度缩放器
    # scaler = GradScaler()

    A = best_loss
    for epoch_index in iterator:
        loss_list = []
        for batch_idx, (input_data, y) in enumerate(train_loader):
            input_data, y = input_data.to(device), y.to(device)

            pred_output = net(input_data)
            loss = criterion(pred_output, y, Flag)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            clip_wights(net)
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
            for split_index in range(split_num):
                split_input = input_data[split_index][0].to(device)
                split_y = y[split_index][0].to(device)

                pred_output = net(split_input)
                loss = criterion(pred_output, split_y, Flag)
                eval_loss.append(loss)
        net.train()
        return sum(eval_loss) / len(eval_loss)


def eval_net(net, eval_loader, result_pic_folder, device):
    net.eval()
    all_info = eval_loader.dataset.all_info
    all_compressor_speed_mae = []
    all_cab_heating_status_act_pos_mae = []
    all_compressor_speed_prb_mae = []
    all_cab_heating_status_act_pos_prb_mae = []
    compressor_speed_r_list = []
    cab_heating_status_act_pos_r_list = []
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
                pred_output = net(split_input)
                # pred_output = net1(split_input)
                # zipped = zip(pred_output1, pred_output2)
                # sums = [x + y for x, y in zipped]
                # pred_output = tuple([s / 2 for s in sums])
                compressor_speed_real = split_y[:, 0]
                cab_heating_status_act_pos_real = split_y[:, 1]

                compressor_speed_pred = pred_output[:,0]
                cab_heating_status_act_pos_pred = pred_output[:,1]

                compressor_speed_mae = torch.mean(torch.abs(compressor_speed_pred - compressor_speed_real))
                compressor_speed_mae_prob = torch.abs(compressor_speed_pred - compressor_speed_real) / torch.clamp(compressor_speed_real,min=0)
                cab_heating_status_act_pos_mae = torch.mean(torch.abs(cab_heating_status_act_pos_pred - cab_heating_status_act_pos_real))
                cab_heating_status_act_pos_mae_prob = torch.abs(cab_heating_status_act_pos_pred - cab_heating_status_act_pos_real) / torch.clamp(cab_heating_status_act_pos_real,min=1)

                compressor_speed_r = np.corrcoef(compressor_speed_pred,compressor_speed_real)[0][1]
                cab_heating_status_act_pos_r = np.corrcoef(cab_heating_status_act_pos_pred,cab_heating_status_act_pos_real)[0][1]

                if np.isnan(compressor_speed_r):
                    data_gap = compressor_speed_pred - compressor_speed_real
                    if (data_gap[0] == data_gap).all():
                        compressor_speed_r = 1
                    else:
                        compressor_speed_r = 0


                if np.isnan(cab_heating_status_act_pos_r):
                    data_gap = cab_heating_status_act_pos_pred - cab_heating_status_act_pos_real
                    if (data_gap[0] == data_gap).all():
                        cab_heating_status_act_pos_r = 1
                    else:
                        cab_heating_status_act_pos_r = 0

                if not np.isnan(compressor_speed_r):
                    compressor_speed_r_list.append(compressor_speed_r)
                if not np.isnan(cab_heating_status_act_pos_r):
                    cab_heating_status_act_pos_r_list.append(cab_heating_status_act_pos_r)

                # 可视化
                series_list = [
                    # [pred_t_r_1, refrigerant_mix_temp],
                    [compressor_speed_pred, compressor_speed_real],
                    [cab_heating_status_act_pos_pred, cab_heating_status_act_pos_real],
                    [compressor_speed_mae_prob, cab_heating_status_act_pos_mae_prob],
                    [split_input[:,0]], [split_input[:,2], split_input[:,4]] , [split_input[:,5]],
                     [split_input[:,8], split_input[:,9], split_input[:,10]]
                ]
                series_name_list = [
                    # ['pred_refrigerant_mix_temp', 'refrigerant_mix_temp'],
                    ['pred_compressor_speed', 'compressor_speed'],
                    ['pred_cab_heating_status_act_pos', 'cab_heating_status_act_pos'],
                    ['pred_compressor_speed_prob','pred_cab_heating_status_act_pos_prob' ],
                    ['ac_pid_out_hp'], ['hi_pressure', 'aim_hi_pressure'], ['exv_pid_iout'],
                     ['temp_p_h_2', 'temp_p_h_5', 'temp_p_h_1_cab_heating']
                ]
                file_name = all_info[batch_idx][split_index].replace('.csv', '').split('/')[-1].split('\\')[-1]
                pic_name = file_name + '.png'
                # print(series_list)
                draw_pred(series_list, series_name_list, result_pic_folder, pic_name)
                # print(pic_name)
                print('file_name', file_name, 'compressor_speed_mae', compressor_speed_mae, 'cab_heating_status_act_pos_mae',
                      cab_heating_status_act_pos_mae, 'compressor_speed_r', compressor_speed_r, 'cab_heating_status_act_pos_r',
                      cab_heating_status_act_pos_r)
                all_compressor_speed_mae.append(compressor_speed_mae.view(1, -1))
                all_cab_heating_status_act_pos_mae.append(cab_heating_status_act_pos_mae.view(1, -1))
                all_compressor_speed_prb_mae.extend(compressor_speed_mae_prob.view(1, -1))
                all_cab_heating_status_act_pos_prb_mae.extend(cab_heating_status_act_pos_mae_prob.view(1, -1))

                # 9+2+2
                all_data_temp = torch.cat((split_input,pred_output,split_y), dim=1)
                all_data.append(all_data_temp.numpy())
            batch_idx += 1


    net.train()
    # np.save('AC_energy_pred/data/data_anyls/all_data1011.npy', all_data)



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
    print('compressor_speed_r_mean',sum(compressor_speed_r_list)/len(compressor_speed_r_list))
    print('cab_heating_status_act_pos_r_mean', sum(cab_heating_status_act_pos_r_list) / len(cab_heating_status_act_pos_r_list))
    print('mae_all_compressor_speed_mae', mae_all_compressor_speed_mae)
    print('mae_all_cab_heating_status_act_pos_mae', mae_all_cab_heating_status_act_pos_mae)
    print('all_compressor_speed_prb_mae', all_compressor_speed_prb_mae)
    print('all_cab_heating_status_act_pos_prb_mae', all_cab_heating_status_act_pos_prb_mae)
    return mae_all_compressor_speed_mae, mae_all_cab_heating_status_act_pos_mae,all_compressor_speed_prb_mae,all_cab_heating_status_act_pos_prb_mae


if __name__ == '__main__':

    parser = my_args()
    args = parser.parse_args()
    device = get_device(device_id=args.device_id)
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

    # data_folder = 'D:/Code/code/submit/eng_test/energy_consumption_control/AC_energy_pred/data/pre_filter/csv_lemans_0110'
    # data_folder = './data/energy_cunsum_data/csv_file'
    data_folder = 'D:/Code/code/submit/eng_test/energy_consumption_control/AC_energy_pred/data/pre_filter/csv_lemans_all_zfq/lemans_5'
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
        train_dataset = CmprControlBaseDataset(train_data_path, True)
        eval_dataset = CmprControlBaseDataset(eval_data_path, False)
        print('train_dataset: ', len(train_dataset))
        print('eval_dataset: ', len(eval_dataset))
        # all_dataset = CmprControlBaseDataset(all_data_path, False)

        # with open(train_dataset_path, 'wb') as f:
        #     pickle.dump(train_dataset, f)
        #
        # with open(eval_dataset_path, 'wb') as f:
        #     pickle.dump(eval_dataset, f)

        # with open(all_dataset_path, 'wb') as f:
        #     pickle.dump(all_dataset, f)

    # 网络

    # net = MLPModel(input_dim=6, output_dim=1)
    net = MyModel()

    if load_pretrain:
        if os.path.exists(pretrain_ckpt_path):
            # net1 = torch.load(pretrain_ckpt_path, map_location=torch.device('cpu'))
            net.load_state_dict(torch.load(pretrain_ckpt_path, map_location=torch.device('cpu')))
    net.to(device)

    # from model.model_cmpr_control import MyModel as MyCab_pos_model
    # net1 = MyCab_pos_model()
    # net1.load_state_dict(torch.load('./data/ckpt/cc_1225_v2.pth', map_location=torch.device('cpu')))
    # net.cab_pos_model.load_state_dict(net1.cab_pos_model.state_dict())
    # torch.save(net.state_dict(), saved_ckpt_path)
    # net.to(torch.float16)

    # net1 = MyModel()
    # net1.load_state_dict(torch.load('./data/ckpt/cc_1108_v2.pth', map_location=torch.device('cpu')))
    # net.cab_pos_model.load_state_dict(net1.cab_pos_model.state_dict())

    # net.load_state_dict(torch.load('./data/ckpt/cc_1105_v1.pth', map_location=torch.device('cpu')))
    # net.load_state_dict(
    # torch.save(net.state_dict(), saved_ckpt_path)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False)
    if Flag_train:
        if Flag == 1:
            optimizer = torch.optim.Adam(net.compressor_speed_model.parameters(),
                                         lr=lr,
                                         weight_decay=weight_decay)
        elif Flag == 2:
            optimizer = torch.optim.Adam(net.cab_pos_model.parameters(),
                                         lr=lr,
                                         weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(net.parameters(),
                                         lr=lr,
                                         weight_decay=weight_decay)

        train_loss_list, eavl_loss_list = train_net(net, train_epoch, train_loader, eval_loader, optimizer, device)
        get_loss_cur(train_loss_list, eavl_loss_list,['train_loss', 'eval_loss'])

    # 测试    eval_batch_size = config.eval_batch_size
    cond_num = len(eval_dataset)

    # eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
    mae_all_compressor_speed_mae, mae_all_cab_heating_status_act_pos_mae, all_compressor_speed_prb_mae, all_cab_heating_status_act_pos_prb_mae = eval_net(
        net, eval_loader, result_pic_folder, device)


