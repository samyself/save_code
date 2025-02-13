import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.array_api import float32


def self_convolution(input_tensor, kernel_size=3):
    """
    对输入张量进行自卷积操作。

    参数:
    - input_tensor: 输入张量，形状为 (batch_size, dim)
    - kernel_size: 卷积核大小，默认为3

    返回:
    - 输出张量，形状为 (batch_size, dim)
    """
    batch_size, dim = input_tensor.shape

    # 将输入张量转换为 (batch_size, 1, dim) 形状
    input_tensor = input_tensor.unsqueeze(1)  # (batch_size, 1, dim)

    # 创建卷积层
    conv_layer = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

    # 使用输入张量作为卷积核
    with torch.no_grad():
        conv_layer.weight[0, 0, :] = input_tensor[0, 0, :kernel_size]

    # 进行卷积操作
    output_tensor = conv_layer(input_tensor)

    # 去除多余的维度
    output_tensor = output_tensor.squeeze(1)  # (batch_size, dim)

    return output_tensor

if __name__ == "__main__":
    # 生成示例数据
    batch_size = 1
    dim = 10
    input_tensor = torch.tensor([[0.0,1.0,2,3,4,5,6,7,8,9]]) # (batch_size, dim)

    print("Input Tensor:")
    print(input_tensor)

    # 进行自卷积
    output_tensor = self_convolution(input_tensor, kernel_size=dim)

    print("Output Tensor after Self-Convolution:")
    print(output_tensor)