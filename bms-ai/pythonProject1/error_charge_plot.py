import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

def merge_data(folder_path,N):
    # 遍历文件夹中的所有文件
    V,C,T,avg_c,avg_v=[],[],[],[],[]
    j=0
    for filename in os.listdir(folder_path):
        j=j+1
        if filename.endswith('.csv') & (j==N):
            file_path = os.path.join(folder_path, filename)
            # 读取单个车辆的CSV文件
            df = pd.read_csv(file_path)
            V = df['Volt']
            C=df['Curr']
            T=df['Temp']
            avg_c = df[['avg_current']]
            avg_v = df[['avg_voltage']]
    return np.array(V), np.array(C),np.array(T),np.array(avg_c),np.array(avg_v)

N=4947
folder_path='D:/data/data_handle_new_test_over75_d0_skip_dot'
Volt,Curr,Temp,avg_current, avg_voltage= merge_data(folder_path,N)

plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.plot(range(len(Volt)),Volt, color="blue", linewidth=1.5, linestyle="-")
plt.title(f"充电电压曲线")
plt.show()

plt.figure()
plt.plot(range(len(Curr)),Curr, color="blue", linewidth=1.5, linestyle="-")
plt.title(f"充电电流曲线")
plt.show()

plt.figure()
plt.plot(range(len(Temp)),Temp, color="blue", linewidth=1.5, linestyle="-")
plt.title(f"充电温度曲线")
plt.show()


df= pd.read_csv('output_data.csv')
j=(df['chargetimes']==N)
error_plot=df['errors'][j]
plt.figure()
plt.plot(range(60,len(error_plot)+60),error_plot, color="blue", linewidth=1.5, linestyle="-")
plt.title(f"SOC估计误差曲线")
plt.show()


def mean_absolute_error(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mape = np.mean(np.abs(actual- predicted))
    return mape

y_test=df['TrueSoc'][j]
predicted_soc=df['preSoc'][j]
mape_value = mean_absolute_error(y_test, predicted_soc)
print(f"平均绝对误差: {mape_value:.2f}%")
plt.figure()
plt.plot(range(60,len(y_test)+60), y_test, color="blue", linewidth=1.5, linestyle="-")
plt.plot(range(60,len(predicted_soc)+60), predicted_soc , color="red", linewidth=1.5, linestyle="-.")
plt.legend(['真实值', '预测值'])
plt.title(f"LSTM回归模型真实值与预测值比对图,平均误差:{mape_value:.2f}%")
plt.show()