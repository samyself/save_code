import torch
import torch.onnx
from model.model_cmpr_control import MyModel
import onnx
import onnx.checker

# 假设 model 是您已经训练好的 PyTorch 模型
PYTORCH_MODEL_PATH = "data/ckpt/cs_1017_v1.pth"

ONNX_MODEL_PATH = "data/ckpt/cs_1017_v122.onnx"

model = MyModel()
model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=torch.device('cpu')))
model.eval()  # 设置模型为评估模式

dummy_input = torch.randn(1, 9)

# 导出 ONNX 文件
try:
    torch.onnx.export(
        model,                # 要导出的模型
        dummy_input,          # 示例输入
        ONNX_MODEL_PATH,      # 输出文件名
        opset_version=11,     # ONNX 操作集版本
        do_constant_folding=True,  # 是否进行常量折叠优化
        input_names=["input"],  # 输入节点名称
        output_names=["output"],  # 输出节点名称
        verbose=True          # 打印详细的导出信息
    )
    print("ONNX model exported successfully.")
except Exception as e:
    print(f"Error exporting ONNX model: {e}")
    exit(1)

# 加载并检查 ONNX 模型
try:
    model_onnx = onnx.load(ONNX_MODEL_PATH)
    onnx.checker.check_model(model_onnx)
    print("ONNX model checked successfully.")
except Exception as e:
    print(f"Error checking ONNX model: {e}")
    exit(1)
