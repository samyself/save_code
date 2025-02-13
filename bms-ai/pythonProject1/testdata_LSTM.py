import numpy as np
import pandas as pd
import tensorflow
import os
from matplotlib import pyplot as plt

def merge_data(folder_path, time_step):
    # 遍历文件夹中的所有文件
    Xs, ys,zs,vs,cs,ts,js,As = [], [], [],[],[],[],[],[]
    j=0
    for filename in os.listdir(folder_path):
        j = j + 1
        if filename.endswith('.csv') & (j>-1):
            file_path = os.path.join(folder_path, filename)
            # 读取单个车辆的CSV文件
            df = pd.read_csv(file_path)
            X = df[['Volt', 'Temp', 'Curr','avg_current','avg_voltage','Ah']]
            y = df['TrueSoc']
            z= df['BmsSoc']
            v = df['Volt']
            c=df['Curr']
            t=df['Temp']
            A=df['Ah']
            for i in range(len(X)-time_step):
                x = X[i:(i + time_step)]
                Xs.append(x)
                ys.append(y[i + time_step])
                zs.append(z[i + time_step])
                vs.append(v[i + time_step])
                cs.append(c[i + time_step])
                ts.append(t[i + time_step])
                js.append(j)
                As.append(A[i + time_step])
    return np.array(Xs), np.array(ys),np.array(zs),np.array(vs),np.array(cs),np.array(ts),np.array(js),np.array(As)

folder_path='D:/data/data_handle_new_test_over70_d0_skip_dot'
time_step = 60
features, labels ,Bmssoc,Volt,Curr,Temp,chargetimes,Ah = merge_data(folder_path,time_step)

print('数据特征信息：', features.shape)
print('标签信息：', labels.shape)

# 数据标准化
#model17/18
#scaler_features_max=3794
#scaler_features_min=5.1
#model16
#scaler_features_max=3799
#scaler_features_min=1.5
#model19
#scaler_features_max=3799
#scaler_features_min=0.5

scaler_features_mean=np.mean(features)
print('scaler_features_mean：', scaler_features_mean)
scaler_features_std=np.std(features)
print('scaler_features_std：', scaler_features_std)

'''
scaled_features=features
print('mean_scaled_features[:,:,0]：', np.mean(scaled_features[:,:,0]))
print('std_scaled_features[:,:,0]：', np.std(scaled_features[:,:,0]))
print('mean_scaled_features[:,:,1]：', np.mean(scaled_features[:,:,1]))
print('std_scaled_features[:,:,1]：', np.std(scaled_features[:,:,1]))
print('mean_scaled_features[:,:,2]：', np.mean(scaled_features[:,:,2]))
print('std_scaled_features[:,:,2]：', np.std(scaled_features[:,:,2]))
print('mean_scaled_features[:,:,3]：', np.mean(scaled_features[:,:,3]))
print('std_scaled_features[:,:,3]：', np.std(scaled_features[:,:,3]))
print('mean_scaled_features[:,:,4]：', np.mean(scaled_features[:,:,4]))
print('std_scaled_features[:,:,4]：', np.std(scaled_features[:,:,4]))
scaled_features[:,:,0]=(scaled_features[:,:,0]-np.mean(scaled_features[:,:,0]))/np.std(scaled_features[:,:,0])
scaled_features[:,:,1]=(scaled_features[:,:,1]-np.mean(scaled_features[:,:,1]))/np.std(scaled_features[:,:,1])
scaled_features[:,:,2]=(scaled_features[:,:,2]-np.mean(scaled_features[:,:,2]))/np.std(scaled_features[:,:,2])
scaled_features[:,:,3]=(scaled_features[:,:,3]-np.mean(scaled_features[:,:,3]))/np.std(scaled_features[:,:,3])
scaled_features[:,:,4]=(scaled_features[:,:,4]-np.mean(scaled_features[:,:,4]))/np.std(scaled_features[:,:,4])
X_train=scaled_features
'''
'''
scaled_features=features
scaled_features[:,:,0]=(scaled_features[:,:,0]-3473.008191046097)/30.792772669392242
scaled_features[:,:,1]=(scaled_features[:,:,1]-40.34848985628858)/3.1367861247564943
scaled_features[:,:,2]=(scaled_features[:,:,2]-100.58347820310509)/51.0371236096011
scaled_features[:,:,3]=(scaled_features[:,:,3]-102.95498699522805)/51.14337598425656
scaled_features[:,:,4]=(scaled_features[:,:,4]-3472.12088796817)/27.946528048566915
X_train=scaled_features
'''

#scaler_features_mean=1437.8032068
#scaler_features_std=1661.94604
X_train=(features-scaler_features_mean)/scaler_features_std
#X_train=(features-scaler_features_min)/(scaler_features_max-scaler_features_min)

# 进行预测
model = tensorflow.keras.models.load_model('lstm_model_25.h5')
predicted_soc = model.predict(X_train)
predicted_soc=predicted_soc*100

'''
file_path = 'output_Temp.csv'
pd.DataFrame(Temp).to_csv(file_path, index=False)
print(f"数组已成功保存到 {file_path}")

file_path = 'output_pre.csv'
pd.DataFrame(predicted_soc).to_csv(file_path, index=False)
print(f"数组已成功保存到 {file_path}")

file_path = 'output_Bmssoc.csv'
pd.DataFrame(Bmssoc).to_csv(file_path, index=False)
print(f"数组已成功保存到 {file_path}")

file_path = 'output_Volt.csv'
pd.DataFrame(Volt).to_csv(file_path, index=False)
print(f"数组已成功保存到 {file_path}")

file_path = 'output_Curr.csv'
pd.DataFrame(Curr).to_csv(file_path, index=False)
print(f"数组已成功保存到 {file_path}")

file_path = 'output_true_soc.csv'
pd.DataFrame(labels).to_csv(file_path, index=False)
print(f"数组已成功保存到 {file_path}")
'''


output = pd.DataFrame({
    "Volt": [],
    "Temp": [],
    "Curr": [],
    "BmsSoc":[],
    "preSoc": [],
    "TrueSoc": [],
    "chargetimes": [],
    "errors":[],
    'Ah':[]
})

errors = predicted_soc[:,0]-labels[:]
#errors=np.abs(errors )
#error=predicted_soc-labels
print('数据特征信息：', Volt.shape)
print('数据特征信息：', labels.shape)
print('数据特征信息：', predicted_soc.shape)
print('数据特征信息：', errors.shape)

output = pd.DataFrame({
    "Volt": Volt,
    "Temp": Temp,
    "Curr": Curr,
    "BmsSoc":Bmssoc,
    "preSoc": predicted_soc[:,0],
    "TrueSoc": labels,
    "chargetimes": chargetimes,
    "errors": errors,
    "Ah": Ah
})
output.to_csv('output_data.csv', index=False)