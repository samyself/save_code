import argparse
import os.path
import pickle

from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import config
from AC_energy_pred.model.my_common import get_loss_cur
from data_utils import get_device
from AC_energy_pred.dataset_ac.dataset_high_press import HighPressBaseDataset
from draw_pic import draw_pred
# from model.model_cmpr_control_tcn import *
from model.model_high_press2 import *
# from model.model_ac_kprate import MyModel as ACKPRate_model

'''
压缩机控制模型
'''
def my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default='2', type=str, help='GPU ids')
    # 是否加载数据集
    parser.add_argument('--load_dataset', default=True, type=bool, help='')
    # 训练数据路径
    parser.add_argument('--train_dataset_path', default='./data/energy_cunsum_pkl/high_press_pkl/high_press_train_dataset_checkpoint800.pkl', type=str, help='')
    # 评测数据路径
    parser.add_argument('--eval_dataset_path', default='./data/energy_cunsum_pkl/high_press_pkl/high_press_eval_dataset_checkpoint800.pkl', type=str, help='')
    # 总体据路径
    parser.add_argument('--all_dataset_path', default='./data/energy_cunsum_pkl/high_press_pkl/high_press_all_dataset_checkpoint800.pkl', type=str, help='')

    # 常修改参数
    # 学习率
    parser.add_argument('--lr', default=1e-5, type=float, help='')
    # 训练batch_size
    parser.add_argument('--train_batch_size', default=1024, type=int, help='')
    # 权重衰减系数
    parser.add_argument('--weight_decay', default=0.01, type=float, help='')
    # 训练步长
    parser.add_argument('--train_epoch', default=100, type=int, help='')
    # 是否加载模型
    parser.add_argument('--load_pretrain', default=False, type=bool, help='')
    # 预训练模型路径
    parser.add_argument('--pretrain_ckpt_path', default='./data/ckpt/high_press/hp_1104_v1.pth', type=str, help='')
    # 模型保存路径
    parser.add_argument('--saved_ckpt_path', default='data/ckpt/high_press/hp_1104_v1.pth', type=str, help='')
    # 是否训练
    parser.add_argument('--Flag_train', default=True, type=bool, help='')
    # 保存图片路径
    parser.add_argument('--result_pic_folder', default='./data/train_result/pic/hp_1104_v1', type=str, help='')
    # 训练时自动保存的测试损失
    parser.add_argument('--best_loss', default=  7.927, type=int, help='')

    return parser
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
        net.Cool_Temp_and_Hi_Press_model.eval()
        loss_list = []
        for batch_idx, (input_data, y) in enumerate(train_loader):
            input_data, y = input_data.to(device), y.to(device)
            pred_output = net(input_data, y[:,0:1], y[:,2:3])[:,1]
            loss = criterion(pred_output, y[:,1])
            optimizer.zero_grad()
            loss.backward()
            # iterator.desc = f"epoch_index:{epoch_index},loss:{loss.item()}"
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
                pred_output = net(split_input, split_y[:,0:1], split_y[:,2:3])[:,1]
                loss = torch.mean(torch.abs(pred_output - split_y[:,1]))
                loss1 = criterion(pred_output, split_y[:,1])
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

                pred_output = net(split_input,split_y[:,0:1], split_y[:,2:3])
                com_out_temp_real = split_y[:,1]

                com_out_temp_pred = pred_output[:,1]

                com_out_temp_mae = torch.mean(torch.abs(com_out_temp_pred - com_out_temp_real))
                loss = criterion(com_out_temp_pred, com_out_temp_real)
                if loss !=  com_out_temp_mae:
                    print('loss != com_out_temp_mae')

                com_out_temp_prob = torch.clamp(torch.abs(com_out_temp_pred - com_out_temp_real) / torch.clamp(com_out_temp_real,min=0),max=1)
                com_out_temp_mae_prob = torch.mean(com_out_temp_prob)
                com_out_temp_r = np.corrcoef(com_out_temp_pred,com_out_temp_real)[0][1]

                if not np.isnan(com_out_temp_r):
                    com_out_temp_r_list.append(com_out_temp_r)

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
    device = get_device(device_id=args.device_id)
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

    data_folder = 'D:/Code/code/submit/energy_consumption_control/AC_energy_pred/data/energy_cunsum_data/csv_high_press_1'
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
    highpress_wight = torch.load('./data/ckpt/high_press/high_press_net_2024-10-9.pth', map_location=torch.device('cpu'))
    net.Cool_Temp_and_Hi_Press_model.load_state_dict(highpress_wight)
    # ak_model = ACKPRate_model()

    # torch.save(net.state_dict(), saved_ckpt_path)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False)

    if Flag_train:
        optimizer = torch.optim.Adam(net.Com_Out_Temp_model.parameters(),
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


