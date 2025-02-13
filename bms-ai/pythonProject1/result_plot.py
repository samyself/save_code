import numpy as np
import pandas as pd
import tensorflow
import os
from matplotlib import pyplot as plt

def merge_data(folder_path, time_step):
    # 遍历文件夹中的所有文件
    Xs, ys = [], []
    j = 0
    for filename in os.listdir(folder_path):
        j = j + 1
        if filename.endswith('.csv') & (j>-1):
            file_path = os.path.join(folder_path, filename)
            # 读取单个车辆的CSV文件
            df = pd.read_csv(file_path)
            X = df[['Volt', 'Temp', 'Curr', 'avg_current', 'avg_voltage']]
            y = df['TrueSoc']
            for i in range(len(X)-time_step):
                x = X[i:(i + time_step)]
                Xs.append(x)
                ys.append(y[i + time_step])
    return np.array(Xs), np.array(ys)

folder_path='D:/data/data_select'
features, labels = merge_data(folder_path,60)

print('数据特征信息：', features.shape)
print('标签信息：', labels.shape)

# 数据标准化
# 数据标准化
scaler_features_mean=np.mean(features)
print('scaler_features_mean：', scaler_features_mean)
scaler_features_std=np.std(features)
print('scaler_features_std：', scaler_features_std)
scaler_features_max=np.max(features)
print('scaler_features_max：', scaler_features_max)
scaler_features_min=np.min(features)
print('scaler_features_min：', scaler_features_min)

#model4
#scaler_features_max=3799
#scaler_features_min=0.3
#model5
#scaler_features_max=3800
#scaler_features_min=8.3
#model16
#scaler_features_max=3799
#scaler_features_min=1.5
#model17
scaler_features_max=3794
scaler_features_min=5.1
X_train=(features-scaler_features_min)/(scaler_features_max-scaler_features_min)
y_train=labels/100
# 加载模型用于预测
model = tensorflow.keras.models.load_model('lstm_model_17.h5')

# 进行预测
predicted_soc = model.predict(X_train)

print('数据特征信息：', predicted_soc.shape)
print('标签信息：', y_train.shape)

# 反归一化以获得真实的SOC值
#predicted_soc = predicted_soc*scaler_labels_std+scaler_labels_mean
#y_train = y_train*(scaler_labels_max-scaler_labels_min)+scaler_labels_min

#predicted_soc=predicted_soc*(scaler_labels_max-scaler_labels_min)+scaler_labels_min

predicted_soc=predicted_soc*100
y_train=y_train*100
# 打印一些结果
print("Predicted SOC:", predicted_soc)
print("True SOC:", y_train)

print('数据特征信息：', predicted_soc.shape)
print('标签信息：', y_train.shape)


# 模型评估
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score

print('**************************输出测试集的模型评估指标结果*******************************')

R=round(r2_score(y_train, predicted_soc), 4)
print('R^2：', round(r2_score(y_train, predicted_soc), 4))
print('均方误差:', round(mean_squared_error(y_train, predicted_soc), 4))
print('解释方差分:', round(explained_variance_score(y_train, predicted_soc), 4))
print('绝对误差:', round(mean_absolute_error(y_train, predicted_soc), 4))



def mean_absolute_error(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mape = np.mean(np.abs(actual- predicted))
    return mape

# 计算 MAPE
mape_value = mean_absolute_error(y_train[1:], predicted_soc[1:,0])
print(f"平均绝对误差: {mape_value:.2f}%")


# 计算误差
errors = predicted_soc[1:,0]-y_train[1:]


#file_path = 'output.csv'
#pd.DataFrame(predicted_soc).to_csv(file_path, index=False)
#print(f"数组已成功保存到 {file_path}")

plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.plot(range(len(y_train)), y_train, color="blue", linewidth=1.5, linestyle="-")
plt.plot(range(len(predicted_soc)), predicted_soc , color="red", linewidth=1.5, linestyle="-.")
plt.legend(['真实值', '预测值'])
#plt.title(f"LSTM回归模型真实值与预测值比对图")
plt.title(f"LSTM回归模型真实值与预测值比对图,平均误差:{mape_value:.2f}%")
plt.show()


# 绘制误差分布图
plt.figure()
plt.hist(errors, bins=20, edgecolor='black')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.show()

# 绘制真实值 vs 预测值的散点图
plt.figure()
plt.scatter(y_train, predicted_soc,alpha=0.5)
plt.title(f"True Values vs Predicted Values,R^2:{R:.2f}")
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linewidth=2)  # 理想情况下通过原点的直线
plt.title(f"True Values vs Predicted Values,R^2:{R:.2f}")
plt.show()