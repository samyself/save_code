import os
import sys

sys.path.append('../AC_energy_pred')
import torch.onnx
from model.model_ac_pid_out import MyModel
import onnx
import onnx.checker
import argparse

def my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aim_model_name', type=str, help='GPU ids')
    parser.add_argument('--pytorch_model_path', type=str, help='GPU ids')
    parser.add_argument('--onnx_model_path', type=str, help='GPU ids')

    return parser

parser = my_args()
args = parser.parse_args()
# 目标模型
aim_model_name = 'model_ac_pid_out_test'
pytorch_model_path = args.pytorch_model_path
onnx_model_path = f'../AC_energy_pred/data/{aim_model_name}.onnx'

"""
    要修改模型名称
"""
net = MyModel()
# net.load_state_dict(torch.load(pytorch_model_path, map_location=torch.device('cpu')))
net.eval()  # 设置模型为评估模式
"""
    要修改模型输入维度
"""
dummy_input = torch.randn(1, 7)

model_scripted = torch.jit.script(net) # Export to TorchScript
model_scripted.save(f'{aim_model_name}_scripted.pt') # Save

# 导出 ONNX 文件
try:
    torch.onnx.export(
        net,  # 要导出的模型
        dummy_input,  # 示例输入
        onnx_model_path,  # 输出文件名
        opset_version=11,  # ONNX 操作集版本
        do_constant_folding=True,  # 是否进行常量折叠优化
        input_names=["input"],  # 输入节点名称
        output_names=["output"],  # 输出节点名称
        verbose=True  # 打印详细的导出信息
    )
    print("ONNX model_ac exported successfully.")
except Exception as e:
    print(f"Error exporting ONNX model_ac: {e}")
    exit(1)

# 加载并检查 ONNX 模型
try:
    model_onnx = onnx.load(onnx_model_path)
    onnx.checker.check_model(model_onnx)
    print("ONNX model_ac checked successfully.")
except Exception as e:
    print(f"Error checking ONNX model_ac: {e}")
    exit(1)
