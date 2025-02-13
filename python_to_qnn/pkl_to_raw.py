import os
import pickle
import sys
import torch
import pandas as pd
sys.path.append('../AC_energy_pred')
# from dataset_ac.dataset_acpid_out import AcPidOutBaseDataset
from tqdm import tqdm
from model_ac.model_ac_pid_out_PID import MyModel
import argparse

def my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dataset_path', type=str, default='../AC_energy_pred/data/pkl/seed2_acpidout_eval_fil_split1023.pkl')
    parser.add_argument('--save_path', type=str, default="../AC_energy_pred/data/ac_1104")
    parser.add_argument('--state_path', type=str, default="../AC_energy_pred/data/ckpt/ac_1022_v1.pth")

    return parser

parser = my_args()
args = parser.parse_args()
eval_dataset_path = args.eval_dataset_path

with open(eval_dataset_path, 'rb') as f:
    eval_dataset = pickle.load(f)

save_path = args.save_path
input_list_save_path = save_path+'/input_data'
input_raw_save_path = save_path+'/input_raw'
output_save_path = save_path+'/output_data'
if not os.path.exists(input_list_save_path):
    os.makedirs(input_list_save_path)
if not os.path.exists(input_raw_save_path):
    os.makedirs(input_raw_save_path)
if not os.path.exists(output_save_path):
    os.makedirs(output_save_path)
"""
    读取自己的模型
"""
net = MyModel()
net.load_state_dict(torch.load(args.state_path,map_location=torch.device('cpu')))
net.eval()
input_list_str= ''
output_list = []
# 保存数据
input_id =0
for i in tqdm(range(1)):
    data = eval_dataset[i]
    for inputs in data[0]:
        for j in range(inputs.shape[0]):
            input = inputs[j,:]
            input_bytes = input.tobytes()
            with open(f'{input_raw_save_path}/input{input_id}.raw', 'wb') as f:
                f.write(input_bytes)
                f.flush()
            input_list_str += f'input{input_id}:={input_raw_save_path}/input{input_id}.raw '
            input_list_str += '\n'
            input_id += 1
            output = net(torch.from_numpy(input))
            output_list.append(output[0,0].tolist())

  # for outputs in data[1]:
  #       for output in outputs:
  #           output_list.append(output)
with open(f'{input_list_save_path}/input_list.txt', 'w') as f:
    f.write(input_list_str)
    f.flush()
output_list = pd.DataFrame(output_list)
output_list.to_csv(f'{output_save_path}/output_list.csv', index=False)
print('pkl to raw done')