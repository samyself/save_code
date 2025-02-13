import torch
import torch.nn as nn
import pandas as pd
from scipy.interpolate import interp1d
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init
import numpy as np

def init_weights_xavier_uniform(layer):
    if type(layer) == nn.LSTM:
        for name, param in layer.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(layer) == nn.Linear:
        init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            layer.bias.data.fill_(0)


class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[256, 128, 64, 32, 16], dropout_val=[0.0,0.4]):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim[0])

        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.linear4 = nn.Linear(hidden_dim[2], hidden_dim[3])
        self.linear5 = nn.Linear(hidden_dim[3], hidden_dim[4])
        self.out_linear = nn.Linear(hidden_dim[4], output_dim)
        self.dp1 = nn.Dropout(dropout_val[0])

        self.bn1 = torch.nn.BatchNorm1d(hidden_dim[0])
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim[1])
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim[2])
        self.bn4 = torch.nn.BatchNorm1d(hidden_dim[3])


    def forward(self, x):
        #第一层
        x1 = self.linear1(x)
        x1 = torch.relu(x1)
        x1 = self.dp1(x1)
        x1 = self.bn1(x1)
        # 第二层
        x2 = self.linear2(x1)
        x2 = torch.relu(x2)
        x2 = self.bn2(x2)
        # 第三层
        x3 = self.linear3(x2)
        x3 = torch.relu(x3)
        x3 = self.bn3(x3)
        # 第四层
        x4 = self.linear4(x3)
        x4 = torch.relu(x4)
        x4 = self.bn4(x4)
        # 第五层
        x5 = self.linear5(x4)

        out = self.out_linear(x5)
        return out

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        # 定义lsmt层
        # batch_first=True表示输入数据的形状是(batch_size, sequence_length, input_size)
        # 而不是默认的(sequence_length, batch_size, input_size)。
        # batch_size是指每个训练批次中包含的样本数量
        # sequence_length是指输入序列的长度
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 定义全连接层，将LSTM层的输出映射到最终的输出空间。
        self.fc = nn.Linear(hidden_size, output_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.mlp = MLPModel(hidden_size,output_size)
        self.mlp.apply(init_weights_xavier_uniform)

        # 初始化LSTM的权重
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)  # 使用Xavier均匀分布初始化输入权重
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)  # 使用正交矩阵初始化隐藏层权重
            elif 'bias' in name:
                init.constant_(param.data, 0)  # 初始化偏置为0

        # 初始化全连接层的权重
        # init.kaiming_uniform_(self.fc.weight.data)  # 使用He均匀分布初始化全连接层权重
        # init.constant_(self.fc.bias.data, 0)  # 初始化全连接层偏置为0

    def forward(self, x):
        # 初始化了隐藏状态h0和细胞状态c0，并将其设为零向量。
        h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)

        # LSTM层前向传播
        # 将输入数据x以及初始化的隐藏状态和细胞状态传入LSTM层
        # 得到输出out和更新后的状态。
        # out的形状为(batch_size, sequence_length, hidden_size)。
        out, _ = self.lstm(x, (h0, c0))
        # out, _ = self.lstm(x)
        # 添加激活函数
        out = torch.relu(out)
        # batch_norm层
        # out = self.bn(out)
        # 全连接层前向传播
        # 使用LSTM层的最后一个时间步的输出out[:, -1, :]（形状为(batch_size, hidden_size)）作为全连接层的输入，得到最终的输出。
        # out = self.fc(out[:, :])

        out = self.mlp(out[:, :])
        return out

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)

class Model():
    def __init__(self, model, loss=None,input_data=None, train_data=None, eval_data=None):
        self.model = model
        self.loss = loss
        self.input_data = input_data
        self.train_data = train_data
        self.eval_data = eval_data


    def model_output(self):
        self.model.eval()
        # 进行前向传播
        with torch.no_grad():  # 不需要梯度计算
            output_data = self.model(self.input_data)
        # 压缩机转速  CEXV开度
        return output_data

    def train(self, train_features, train_labels, test_features, test_labels,
              num_epochs, learning_rate, weight_decay, batch_size):
        device = 'cpu'
        net = self.model
        loss = self.loss
        train_ls, test_ls = [], []

        train_iter = load_array((train_features, train_labels), batch_size, True)


        # 这里使用的是Adam优化算法
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        for epoch in tqdm(range(num_epochs)):
            loss_list = []
            for X, y in train_iter:
                optimizer.zero_grad()
                l = loss(net(X), y)
                l.backward()
                optimizer.step()
                loss_list.append(l.tolist())
            train_ls.append(sum(loss_list)/len(loss_list))
            print(f'train_loss:epoch{epoch+1}',train_ls[-1])
            if test_labels is not None:
                test_ls.append((loss(self.model(test_features), test_labels)).tolist())
                print(f'test_loss:epoch{epoch+1}',test_ls[-1])
        return train_ls, test_ls



loss_MSE = nn.MSELoss(reduction='mean')
def loss (y_hat, y):
    res = torch.sqrt(loss_MSE(y_hat,y))
    return res

def load_csv(file_path):
    df=pd.read_csv(file_path)
    data=df.to_dict('records')
    return data


def get_loss_cur(first, second, name, save_name=None):
    # 创建一个新的图形
    plt.figure()
    # 绘制训练损失曲线
    plt.plot(first, label=f'{name[0]}')
    # 绘制测试损失曲线
    plt.plot(second, label=f'{name[1]}')
    # 添加图例
    plt.legend()
    # 显示网格
    plt.grid(True)
    # 展示图表
    plt.show()

    # plt.savefig(save_name)

def file_data_in(file_path,file_name_list):
    input_data = []
    if file_name_list == 'all':
        file_name_list = os.listdir(file_path)
        # file_name_list = [os.path.join(file_path, file_name) for file_name in all_file_name]

    for file_name in file_name_list:
        input_data_dir = os.path.join(file_path, file_name)
        input_data.extend(load_csv(input_data_dir))
    return input_data

#以dim进行归一化
def come_1(data_tensor,dim=0):
    min_vals,_ = torch.min(data_tensor,dim=dim)
    max_vals,_ = torch.max(data_tensor,dim=dim)

    # 进行最小-最大归一化
    normalized_data = (data_tensor - min_vals) / (max_vals - min_vals)
    return normalized_data

def data_set(input_list,feature_dim):
    all_data = torch.tensor(input_list)
    # loss_list = []

    L = all_data.shape[0]

    #随机抽样
    train_num = int(L/5*4)
    torch.manual_seed(42)
    random_permutation = torch.randperm(L)

    train_data_index = random_permutation[:train_num]
    test_data_index = random_permutation[train_num:]
    train_data = all_data[train_data_index]
    test_data = all_data[test_data_index]

    # train_data = all_data[:int(L/5*4),:]
    # test_data = all_data[int(L/5*4):-1,:]

    train_features = train_data[:,:feature_dim]
    # train_features = come_1(train_features)
    train_labels = train_data[:,feature_dim:]
    test_features = test_data[:,:feature_dim]
    # test_features = come_1(test_features)
    test_labels = test_data[:, feature_dim:]
    return train_features,train_labels,test_features,test_labels


def draw_pred(series_list, series_name_list, pic_folder, pic_name):
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    total_sub_pic_num = len(series_list)
    plt.figure(figsize=(15, total_sub_pic_num * 6))

    for pic_index in range(total_sub_pic_num):
        sub_pic_data = series_list[pic_index]
        sub_pic_name = series_name_list[pic_index]
        plt.subplot(total_sub_pic_num, 1, pic_index + 1)
        for sub_index in range(len(sub_pic_name)):
            data = sub_pic_data[sub_index]
            name = sub_pic_name[sub_index]
            x = np.arange(len(data))
            plt.plot(x, data, label=name)
            plt.legend()

    if not os.path.exists(pic_folder):
        os.makedirs(pic_folder)
    pic_path = os.path.join(pic_folder, pic_name)

    plt.tight_layout(pad=1.08)
    plt.savefig(pic_path)
    plt.close()


def save_csv(data, file_path):
    data.to_csv(file_path, index=False)
    print('csv文件已保存至：', file_path)

# 字典列表转为列表字典
def dictlist_to_listdict(dict_list):
    keys = dict_list[0].keys()
    # 使用字典推导式进行转换
    list_dict = {key: [d[key] for d in dict_list] for key in keys}
    return list_dict

# 创建一个函数来生成和显示直方图
def create_histogram(data, step= 1):
    max_value = max(data)
    min_value = min(data)
    bins = [i for i in range(int(min_value), int(max_value) + 1 + step, step)]
    res = plt.hist(data, bins=bins, edgecolor='black')
    hist = res[0]
    bin_edges = res[1]
    # 打印统计结果
    for i in range(len(hist)):
        print(f"{bin_edges[i]} - {bin_edges[i + 1]}: {int(hist[i])}")
    # print(f"{bin_edges[-1]} - {bin_edges[-1] + 1}: {int(hist[-1])}")
    # 绘制直方图
    plt.xlabel('Value Range')
    plt.ylabel('Frequency')
    plt.title('Histogram of Number Counts in Ranges')
    plt.xticks(bin_edges)
    plt.show()

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.5):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        y = self.relu(out + res)
        # 检查中间结果是否包含 NaN
        if torch.isnan(y).any():
            print("NaN detected in TemporalBlock output.")



        return y


# TCN网络
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.001):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        y = self.network(x)
        # 在 TCN 输出后也检查是否包含 NaN
        assert not torch.isnan(y).any(), "NaN detected in TCN output."
        return y



if __name__ == '__main__':
    file_path = '../data/myhcsv/all_csv'
    file_name_list = ['2023-11-30 16_23_06_MS11_THEM_A_E4U3_T151, V015_Amb-7~-15℃_2人_后排关闭_40kph_行车加热_座椅加热(无感)_能耗测试.csv',
                      '2023-12-04 12_15_31_THEM_A_E4U3_T153_V015_-10℃~-7℃晴天冷启动高速&低速工况测试.csv'
                      ]
    data = file_data_in(file_path,file_name_list)

