#!/bin/bash
QNN_SDK_ROOT="/home/mi/QNN/qairt/2.26.0.240828"
export PYTHONPATH=${QNN_SDK_ROOT}/lib/python:$PYTHONPATH

export ANDROID_NDK_ROOT="android-ndk-r26c-linux.zip"
export PATH=${ANDROID_NDK_ROOT}:${PATH}
export PATH=/usr/bin/aarch64-linux-gnu:${PATH}
#export QNN_TARGET_ARCH=aarch64-android

# 模型路径

WORK_PATH="/home/mi/Project/energy_consumption_control/AC_energy_pred"
cd "$WORK_PATH"
echo "Current directory: $(pwd)"

#保存的根路径
save_path="./data/ac_1104"

#保存时的模型名称
model_name="model_ac_1104"


# 1. 确保原始开发的模型中各种算子是可用的（torch.onn.export()）
# 输入.pth文件，导出ONNX模型  ，进入onnx_python_myh.py中修改模型名称与模型输入维度
#保存.pt模型名称
MODEL_NAME='CompressorSpeedPIDModel'
#原始模型路径
PYTORCH_MODEL_PATH="./data/ckpt/ac_1022_v1.pth"
# 保存的ONNX模型的路径
if test -e "${save_path}/onnx_model"; then
  echo "onnx_model folder exists."
else
  mkdir -p "${save_path}/onnx_model"
fi
ONNX_MODEL_PATH="${save_path}/onnx_model/${model_name}.onnx"

if test -e "$ONNX_MODEL_PATH"; then
  echo "ONNX model already exists. Skipping conversion."
else
  echo "Converting PyTorch model to ONNX format..."
  python ../python_to_qnn/onnx_python_myh.py \
  --aim_model_name $MODEL_NAME \
  --pytorch_model_path $PYTORCH_MODEL_PATH \
  --onnx_model_path $ONNX_MODEL_PATH
fi
echo "Converting PyTorch model successfully."


# 2. 利用QNN对应的工具进行转换（onnx->cpp bin-->.so）
# 转换ONNX模型
echo "Converting ONNX model to QNN format..."
cpp_model_folder="cpp/"
aim_type="x86_64-linux-clang"
#保存CPP的路径
if test -e "${save_path}/cpp"; then
  echo "cpp folder exists."
else
  mkdir -p "${save_path}/cpp"
fi
CPP_MODEL_PATH="${save_path}/cpp/${model_name}.cpp"
BIN_MODEL_PATH="${save_path}/cpp/${model_name}.bin"
# 转化为QNN格式
python ${QNN_SDK_ROOT}/bin/${aim_type}/qnn-onnx-converter \
-i "$ONNX_MODEL_PATH" \
-o "$CPP_MODEL_PATH"

echo "Converting QNN model to CPP successfully."
#模型编译
# 保存编译文件.so的路径 要修改为libmodel_{原本的.pth名字}
model_qnn_file_name="lib${model_name}.so"
if test -e "${save_path}/model_libs"; then
  echo "model_libs folder exists."
else
  mkdir -p "${save_path}/model_libs"
fi
QNN_MODEL_FOLDER="${save_path}/model_libs/bin"
QNN_MODEL_PATH="${save_path}/model_libs/bin/${aim_type}/${model_qnn_file_name}"

python ${QNN_SDK_ROOT}/bin/${aim_type}/qnn-model-lib-generator \
  -c "$CPP_MODEL_PATH" \
  -b "$BIN_MODEL_PATH" \
  -o "$QNN_MODEL_FOLDER" \
  -t $aim_type


if [ $? -ne 0 ]; then
    echo "Error converting ONNX model to QNN format."
    exit 1
else
    echo "ONNX model converted to QNN format successfully."
fi

# 3. 校验转换后的模型与原始Python之间的结果误差
# 运行原生PyTorch模型
echo "Running PyTorch model..."
python ../python_to_qnn/pkl_to_raw.py --eval_dataset_path "./data/pkl/seed2_acpidout_eval_fil_split1023.pkl" \
--save_path $save_path \
--state_path $PYTORCH_MODEL_PATH

# 运行QNN模型
echo "Running QNN model..."
QNN_OUTPUT_PATH=${save_path}/qnn_output
if test -e "${QNN_OUTPUT_PATH}"; then
  echo "QNN_OUTPUT_PATH folder exists."
else
  mkdir -p "${QNN_OUTPUT_PATH}"
fi

#sudo chmod 777 ${QNN_SDK_ROOT}/bin/${aim_type}/qnn-net-run
${QNN_SDK_ROOT}/bin/${aim_type}/qnn-net-run \
      --model ${QNN_MODEL_PATH} \
      --input_list ${save_path}/input_data/input_list.txt \
      --backend ${QNN_SDK_ROOT}/lib/${aim_type}/libQnnCpu.so \
      --output_dir ${QNN_OUTPUT_PATH}


if [ $? -ne 0 ]; then
    echo "Error running QNN model."
    exit 1
else
    echo "QNN model ran successfully."
fi

# 比较结果：输入为QNN模型的推理结果和pytorch输出对比的pic，路径为${save_path}/pic，并计算mae
echo "Comparing results..."
CMP_PIC_PATH=${save_path}/pic
if test -e "${CMP_PIC_PATH}"; then
  echo "CMP_PIC_PATH folder exists."
else
  mkdir -p "${CMP_PIC_PATH}"
fi

python ../python_to_qnn/compare_mean_test1.py \
--qnn_res_path ${QNN_OUTPUT_PATH} \
--save_path ${save_path} \
--pic_path ${CMP_PIC_PATH}

echo "All tasks completed successfully."
