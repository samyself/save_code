import argparse
import os.path
import pickle

from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import config
from AC_energy_pred.model.my_common import get_loss_cur
# from data_utils import get_device
from dataset_ac.dataset_high_press import HighPressBaseDataset
from draw_pic import draw_pred
# from model.model_cmpr_control_tcn import *
from model.model_high_press1 import *
# from model.model_ac_kprate import MyModel as ACKPRate_model


'''
用于训练+测试排气温度
'''
def my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default='2', type=str, help='GPU ids')
    # 是否加载数据集
    parser.add_argument('--load_dataset', default=False, type=bool, help='')
    # 训练数据路径
    parser.add_argument('--train_dataset_path', default='./data/energy_cunsum_pkl/high_press_pkl/hp_train_666.pkl', type=str, help='')
    # 评测数据路径  hp_test_777_test  hp_eval_777
    parser.add_argument('--eval_dataset_path', default='./data/energy_cunsum_pkl/high_press_pkl/hp_eval_666.pkl', type=str, help='')
    # 总体据路径
    parser.add_argument('--all_dataset_path', default='./data/energy_cunsum_pkl/high_press_pkl/high_press_all_dataset_checkpoint800_filter.pkl', type=str, help='')

    # 常修改参数
    # 学习率
    parser.add_argument('--lr', default=1e-2, type=float, help='')
    # 训练batch_size
    parser.add_argument('--train_batch_size', default=1024, type=int, help='')
    # 权重衰减系数
    parser.add_argument('--weight_decay', default=0.01, type=float, help='')
    # 训练步长
    parser.add_argument('--train_epoch', default=20, type=int, help='')
    # 是否加载模型
    parser.add_argument('--load_pretrain', default=True, type=bool, help='')
    # 预训练模型路径
    parser.add_argument('--pretrain_ckpt_path', default='./data/ckpt/high_press/hp_20250210_v1.pth', type=str, help='')
    # 模型保存路径
    parser.add_argument('--saved_ckpt_path', default='data/ckpt/high_press/hp_20250210_v1.pth', type=str, help='')
    # 是否训练
    parser.add_argument('--Flag_train', default=False, type=bool, help='')
    # 保存图片路径
    parser.add_argument('--result_pic_folder', default='./data/train_result/pic/hp_20250210_v1', type=str, help='')
    # 训练时自动保存的测试损失
    parser.add_argument('--best_loss', default= 6, type=int, help='')


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
    if len(output.shape) == 2:
        output = output[:,0]
    if len(target.shape) == 2:
        target = target[:,0]
    # mae_loss
    l1_loss = nn.L1Loss(reduction='mean')

    # mse_loss
    # l2_loss = nn.MSELoss(reduction='mean')
    com_out_temp_r = XiangGuanDu(output, target)
    # com_out_temp_r1 = np.corrcoef(output.detach().numpy(), target.detach().numpy())[0][1]

    # max_lim_loss = torch.mean(torch.relu(output - 110)) + 1

    total_loss = l1_loss(output , target) + 3*torch.exp((1 - com_out_temp_r)**1)
    # total_loss = l1_loss(output , target) + torch.exp((1 - com_out_temp_r)**1)**2
    # total_loss = l1_loss(output , target)

    return total_loss


def train_net(net, train_epoch, train_loader, eval_loader, optimizer, device):
    criterion = custom_loss
    # criterion = nn.MSELoss(reduction='mean')
    train_loss = []
    eval_loss = []
    iterator = tqdm(range(train_epoch))

    A = best_loss
    for epoch_index in iterator:
        net.Cool_Temp_and_Hi_Press_model.eval()
        loss_list = []
        for batch_idx, (input_data, y) in enumerate(train_loader):
            input_data, y = input_data.to(device), y.to(device)
            pred_output = net(input_data)

            # loss1 = criterion(pred_output[:, 0], y[:, 0])
            loss2 = criterion(pred_output[:,1], y[:,1])
            # loss3 = criterion(pred_output[:, 2], y[:, 2])
            loss = loss2

            optimizer.zero_grad()
            loss.backward()
            # iterator.desc = f"epoch_index:{epoch_index},loss:{loss.item()}"
            # torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)
            optimizer.step()
            # clip_wights(net.Com_Out_Temp_model)
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
                split_input = input_data[split_index][0].to(device)
                split_y = y[split_index][0].to(device)
                # mask = (split_y[:, 1] > 115) | (split_y[:, 1] < 40)
                # if mask.any():
                    # print('mask')

                split_len = len(split_input)
                # pred_output = torch.zeros_like(split_y)
                # for i in range(split_len):
                #     if i % 30 == 0:
                #         net.last_ac_pid_out_hp = split_input[i,0].unsqueeze(0)
                #
                #     pred_output[i] = net(split_input[i])
                #     # net.last_ac_pid_out_hp = pred_output[i]

                # 多轮
                pred_output = net(split_input)[:,1]
                # loss = torch.mean(torch.abs(pred_output - split_y[:,1]))
                loss = criterion(pred_output, split_y[:,1])
                eval_loss.append(loss.item())

            # eval_loss.append(sum(loss_list) / len(loss_list))
        net.train()
        return sum(eval_loss) / len(eval_loss)


def eval_net(net, eval_loader, result_pic_folder, device):
    net.eval()
    criterion = custom_loss
    all_info = eval_loader.dataset.all_info
    all_com_out_temp_mae = []
    all_com_out_temp_prb_mae = []
    com_out_temp_r_list = []

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
                # mask = (split_y[:,1] > 115) | (split_y[:,1] < 40)
                # if mask.any():
                #     print('mask')

                # split_len = len(split_input)
                # pred_output = torch.zeros_like(split_y)
                # for i in range(split_len):
                #     new_split_input = split_input[i].clone()
                #     if i  == 0:
                #         # net.last_ac_pid_out_hp = split_input[i,0].unsqueeze(0)
                #         pass
                #     else:
                #         new_split_input[0] = pred_output[i - 1].detach()

                    # pred_output[i] = net(new_split_input)
                    # net.last_ac_pid_out_hp = pred_output[i]

                pred_output = net(split_input)
                com_out_temp_real = split_y[:,1]

                com_out_temp_pred = pred_output[:,1]

                com_out_temp_mae = torch.mean(torch.abs(com_out_temp_pred - com_out_temp_real))
                # loss = criterion(com_out_temp_pred, com_out_temp_real)
                # if loss !=  com_out_temp_mae:
                #     print('loss != com_out_temp_mae')

                com_out_temp_prob = torch.abs((com_out_temp_pred - com_out_temp_real)/ com_out_temp_real)
                com_out_temp_mae_prob = torch.mean(com_out_temp_prob)
                com_out_temp_r = np.corrcoef(com_out_temp_pred,com_out_temp_real)[0][1]

                if not np.isnan(com_out_temp_r):
                    com_out_temp_r_list.append(com_out_temp_r)
                else:
                    com_out_temp_r_list.append(0.0)

                # 可视化
                series_list = [
                    [com_out_temp_pred, com_out_temp_real],
                    [com_out_temp_prob]
                ]
                series_name_list = [
                    # ['pred_refrigerant_mix_temp', 'refrigerant_mix_temp'],
                    ['com_out_temp_pred', 'com_out_temp_real'],
                    ['com_out_temp_mae_prob']
                ]
                file_name = all_info[batch_idx][split_index].replace('.csv', '').split('/')[-1].split('\\')[-1]
                pic_name = file_name + '.png'
                # print(series_list)
                draw_pred(series_list, series_name_list, result_pic_folder, pic_name)
                # print(pic_name)
                print('file_name', file_name, 'com_out_temp_mae', com_out_temp_mae,'com_out_temp_r', com_out_temp_r,'com_out_temp_mape',com_out_temp_mae_prob)
                all_com_out_temp_mae.append(com_out_temp_mae)
                all_com_out_temp_prb_mae.append(com_out_temp_mae_prob)

                # 9+2+2
                # all_data_temp = torch.cat((split_input,pred_output,split_y), dim=1)
                # all_data.append(all_data_temp.numpy())
            batch_idx += 1


    net.train()
    # np.save('AC_energy_pred/data/data_anyls/all_data1011.npy', all_data)

    # 计算指标
    # all_refrigerant_mix_temp_mae = torch.cat(all_refrigerant_mix_temp_mae)
    mae_all_com_out_temp_mae = sum(all_com_out_temp_mae)/len(all_com_out_temp_mae)
    mae_all_com_out_temp_prb_mae = torch.mean(torch.tensor(all_com_out_temp_prb_mae))
    mae_com_out_temp_r_list = sum(com_out_temp_r_list)/len(com_out_temp_r_list)


    print('mae_all_com_out_temp_mae', mae_all_com_out_temp_mae)
    print('mae_all_com_out_temp_prb_mae', mae_all_com_out_temp_prb_mae)
    print('mae_com_out_temp_r_list', mae_com_out_temp_r_list)
    return mae_all_com_out_temp_mae, mae_all_com_out_temp_prb_mae,mae_com_out_temp_r_list


if __name__ == '__main__':

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
    # 是否训练
    Flag_train = args.Flag_train
    result_pic_folder = args.result_pic_folder
    # 训练时自动保存的测试损失
    best_loss = args.best_loss

    if not os.path.exists(result_pic_folder):
        os.makedirs(result_pic_folder)

    data_folder = 'D:/Code/code/submit/energy_consumption_control/AC_energy_pred/data/energy_cunsum_data/hp_file/csv_lemans_5'
    # data_folder = 'D:/Code/code/submit/energy_consumption_control/AC_energy_pred/data/energy_cunsum_data/hp_file/hp_test'
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
    seed = 1
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
        train_dataset = HighPressBaseDataset(train_data_path, True)
        eval_dataset = HighPressBaseDataset(eval_data_path, False)
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
    net.to(device)
    highpress_wight = torch.load('./data/ckpt/high_press/hp_0207_v4_avg_5.pth', map_location=torch.device('cpu'))
    net.Cool_Temp_and_Hi_Press_model.load_state_dict(highpress_wight)
    # ak_model = ACKPRate_model()

    # torch.save(net.state_dict(), saved_ckpt_path)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False)

    if Flag_train:
        optimizer = torch.optim.Adam(net.Com_Out_Temp_model.parameters(),
        # optimizer=torch.optim.Adam(net.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)

        train_loss_list, eavl_loss_list = train_net(net, train_epoch, train_loader, eval_loader, optimizer, device)
        get_loss_cur(train_loss_list, eavl_loss_list,['train_loss', 'eval_loss'])

    # torch.save(net.state_dict(), saved_ckpt_path)
    # 测试
    eval_batch_size = config.eval_batch_size
    cond_num = len(eval_dataset)

    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
    mae_all_ac_pid_out_mae, mae_all_ac_pid_out_prb_mae,mae_ac_pid_out_r_list = eval_net(
        net, eval_loader, result_pic_folder, device)


