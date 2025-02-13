#汇总模型
from model.model_cmpr_control import MyModel
# 转速模型
from model.com_speed_model_he import MyModel as CSModel
# 开度模型
from model.cab_pos_model_he import MyModel as CPModel
import torch


if __name__ == '__main__':
    # com_speed_ckpt_path = 'data/ckpt/cc_1011_v3.pth'
    # cab_pos_ckpt_path = 'data/ckpt/cc_1011_v7.pth'
    # saved_ckpt_path = 'data/ckpt/com_control_model1012.pth'

    com_speed_ckpt_path = './data/ckpt/cc_1025_v1.pth'
    cab_pos_ckpt_path = './data/ckpt/cc_1025_v1_2.pth'
    saved_ckpt_path = 'data/ckpt/com_control_model1025.pth'

    net = MyModel()
    com_speed_net = CSModel()
    cab_pos_net = CPModel()

    com_speed_net.load_state_dict(torch.load(com_speed_ckpt_path))
    cab_pos_net.load_state_dict(torch.load(cab_pos_ckpt_path))

    try:
        net.compressor_speed_model.load_state_dict(com_speed_net.compressor_speed_model.state_dict())
        print('compressor_speed_model loaded')
    except:
        print('compressor_speed_model not loaded')

    try:
        net.cab_pos_model.load_state_dict(cab_pos_net.cab_pos_model.state_dict())
        print('cab_pos_model loaded')
    except:
        print('cab_pos_model not loaded')

    torch.save(net.state_dict(), saved_ckpt_path)
