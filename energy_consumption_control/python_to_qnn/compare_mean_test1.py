import pandas as pd
import numpy as np
from draw_pic import draw_pred
import torch
import argparse

def my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qnn_res_path', type=str, default='../AC_energy_pred/data/ac_1104/qnn_output')
    parser.add_argument('--save_path', type=str, default='../AC_energy_pred/data/ac_1104')
    parser.add_argument('--pic_path', type=str, default='../AC_energy_pred/data/ac_1104/pic')

    return parser

parser = my_args()
args = parser.parse_args()
save_path = args.save_path
# eval_dataset_path = args.eval_dataset_path
pytorch_output_path = save_path+'/output_data'+'/output_list.csv'

pytorch_output = pd.read_csv(pytorch_output_path)
pytorch_output = pytorch_output.values[:,0].tolist()
print('pytorch_output is list,len=',len(pytorch_output))

qnn_output_file = args.qnn_res_path

qnn_output = []
for i in range(len(pytorch_output)):
    qnn_output_file_path = f"{qnn_output_file}/Result_{i}/output.raw"
    data = np.fromfile(qnn_output_file_path, dtype=np.float32).tolist()
    qnn_output.extend(data)
print('qnn_output is list,len=',len(qnn_output))

series_list=[[pytorch_output,qnn_output]]
series_name_list = [['pytorch', 'qnn']]
result_pic_folder = args.pic_path
file_name = 'result_output'
pic_name = file_name + '.png'
draw_pred(series_list, series_name_list, result_pic_folder, pic_name)

mae = torch.abs(torch.tensor(pytorch_output) - torch.tensor(qnn_output)).mean()
print('mae is ',mae)







