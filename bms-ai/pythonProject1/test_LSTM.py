import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,GlobalAveragePooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


def merge_data(folder_path, time_step):
    # 遍历文件夹中的所有文件
    Xs, ys = [], []
    j=0
    for filename in os.listdir(folder_path):
        j = j + 1
        if filename.endswith('.csv') & (j>-1):
            file_path = os.path.join(folder_path, filename)
            # 读取单个车辆的CSV文件
            df = pd.read_csv(file_path)
            X = df[['Volt', 'Temp', 'Curr','avg_current','avg_voltage','Ah']]
            y = df['TrueSoc']
            for i in range(len(X)-time_step):
                x = X[i:(i + time_step)]
                Xs.append(x)
                ys.append(y[i + time_step])
    return np.array(Xs), np.array(ys)

folder_path='D:/data/data_handle_SOC_over70_d0_skip_dot'
time_step = 60
features, labels = merge_data(folder_path,time_step)

print('数据特征信息：', features.shape)
print('标签信息：', labels.shape)

'''
scaled_features=features
print('max_scaled_features[:,:,0]：', np.max(scaled_features[:,:,0]))
print('max_scaled_features[:,:,1]：', np.max(scaled_features[:,:,1]))
print('max_scaled_features[:,:,2]：', np.max(scaled_features[:,:,2]))
print('min_scaled_features[:,:,0]：', np.min(scaled_features[:,:,0]))
print('min_scaled_features[:,:,1]：', np.min(scaled_features[:,:,1]))
print('min_scaled_features[:,:,2]：', np.min(scaled_features[:,:,2]))
scaled_features[:,:,0]=(scaled_features[:,:,0]-np.min(scaled_features[:,:,0]))/(np.max(scaled_features[:,:,0])-np.min(scaled_features[:,:,0]))
scaled_features[:,:,1]=(scaled_features[:,:,1]-np.min(scaled_features[:,:,1]))/(np.max(scaled_features[:,:,1])-np.min(scaled_features[:,:,1]))
scaled_features[:,:,2]=(scaled_features[:,:,2]-np.min(scaled_features[:,:,2]))/(np.max(scaled_features[:,:,2])-np.min(scaled_features[:,:,2]))
scaled_features[:,:,3]=(scaled_features[:,:,3]-np.min(scaled_features[:,:,2]))/(np.max(scaled_features[:,:,2])-np.min(scaled_features[:,:,2]))
scaled_features[:,:,4]=(scaled_features[:,:,4]-np.min(scaled_features[:,:,0]))/(np.max(scaled_features[:,:,0])-np.min(scaled_features[:,:,0]))
'''

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
print('mean_scaled_features[:,:,5]：', np.mean(scaled_features[:,:,5]))
print('std_scaled_features[:,:,5]：', np.std(scaled_features[:,:,5]))
scaled_features[:,:,0]=(scaled_features[:,:,0]-np.mean(scaled_features[:,:,0]))/np.std(scaled_features[:,:,0])
scaled_features[:,:,1]=(scaled_features[:,:,1]-np.mean(scaled_features[:,:,1]))/np.std(scaled_features[:,:,1])
scaled_features[:,:,2]=(scaled_features[:,:,2]-np.mean(scaled_features[:,:,2]))/np.std(scaled_features[:,:,2])
scaled_features[:,:,3]=(scaled_features[:,:,3]-np.mean(scaled_features[:,:,3]))/np.std(scaled_features[:,:,3])
scaled_features[:,:,4]=(scaled_features[:,:,4]-np.mean(scaled_features[:,:,4]))/np.std(scaled_features[:,:,4])
scaled_features[:,:,5]=(scaled_features[:,:,5]-np.mean(scaled_features[:,:,5]))/np.std(scaled_features[:,:,5])
'''

# 数据标准化
scaler_features_mean=np.mean(features)
print('scaler_features_mean：', scaler_features_mean)
scaler_features_std=np.std(features)
print('scaler_features_std：', scaler_features_std)
scaler_features_max=np.max(features)
print('scaler_features_max：', scaler_features_max)
scaler_features_min=np.min(features)
print('scaler_features_min：', scaler_features_min)


scaler_labels_max=np.max(labels)
print('scaler_labels_max：', scaler_labels_max)
scaler_labels_min=np.min(labels)
print('scaler_labels_min：', scaler_labels_min)
scaler_labels_mean=np.mean(labels)
print('scaler_labels_mean：', scaler_labels_mean)
scaler_labels_std=np.std(labels)
print('scaler_labels_std：', scaler_labels_std)

scaled_features=(features-scaler_features_mean)/scaler_features_std
#scaled_labels=(labels-scaler_labels_mean)/scaler_labels_std
#scaled_features=(features-scaler_features_min)/(scaler_features_max-scaler_features_min)
scaled_labels=labels/100
#scaled_labels=(labels-scaler_labels_min)/(scaler_labels_max-scaler_labels_min)
print('scaled_labels：', scaled_labels.shape)
print('scaled_features：', scaled_features.shape)
# 划分训练集和测试集（8:2）
#X_train=scaled_features
#y_train=scaled_labels
X_train, X_test, y_train, y_test = train_test_split(scaled_features, scaled_labels , test_size=0.2, random_state=42)
'''
train_size = int(len(scaled_features) * 0.8)
test_size = len(scaled_features) - train_size
X_train= scaled_features[0:train_size]
X_test= scaled_features[train_size:len(scaled_features)]
y_train=scaled_labels[0:train_size]
y_test= scaled_labels[train_size:len(scaled_features)]
'''

# 输出拆分后的训练集和测试集样本形状

print('训练集特征信息：', X_train.shape)
print('训练集标签信息：', y_train.shape)
print('测试集特征信息：', X_test.shape)
print('测试集标签信息：', y_test.shape)

#print('y_test:',y_test)

# 调整输入数据的维度以便于LSTM层接收
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

print('训练集增加维度后的特征信息：', X_train.shape, '训练集增加维度后的标签信息：', y_train.shape)
print('测试集增加维度后的特征信息：', X_test.shape, '测试集增加维度后的标签信息：', y_test.shape)
print(X_train.shape[0])
print(X_train.shape[1])
print(X_train.shape[2])



'''
# 构建模型
model = Sequential()
# CNN层
model.add(Conv2D(16, kernel_size=(3, 3),activation='relu', input_shape=(X_train.shape[1], X_train.shape[2],1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# LSTM层
model.add(Reshape((60,-1)))  # 展平成一维向量
model.add(LSTM(32, return_sequences=False))

# 全连接层
model.add(Dense(1, activation='linear'))

# 编译模型
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# 打印模型摘要
model.summary()

# 训练模型
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
'''
# 定义学习率调度器
def lr_scheduler(epoch, lr):
    initial_lr = 0.001
    decay_factor = 0.999
    decayed_lr = initial_lr * (decay_factor ** epoch)
    return max(decayed_lr, 0.00001)  # 确保学习率不低于某个阈值

# 使用回调函数记录误差
class LossHistory(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()  # 构建序惯模型
#model.add(layers.LSTM(300, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),unit_forget_bias=True))# LSTM层
model.add(layers.LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))# LSTM层
model.add(layers.LSTM(16, return_sequences=False))# LSTM层
# model.add(GlobalAveragePooling1D())
# model.add(Dense(32, activation='tanh'))
# model.add(Dense(16, activation='tanh'))
model.add(layers.Dense(1))  # 输出层
#编译模型
lr_reducer = tensorflow.keras.callbacks.LearningRateScheduler(lr_scheduler)
opt = Adam()
#model.compile(optimizer=opt, loss='mse', metrics=['mae'])

#opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)  # 优化器
model.compile(loss='mean_squared_error', optimizer=opt)  # 编译c

# 训练模型
EarlyStopper = EarlyStopping(patience=4, monitor='loss', mode='min')
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test),callbacks=[lr_reducer,EarlyStopper])

model.save('lstm_model_25.h5')


'''
history = MetricsHistory()

# 创建一个学习率变化器
lr_reducer = tensorflow.keras.callbacks.LearningRateScheduler(lr_scheduler)

# 创建一个 EarlyStopping 回调函数来防止过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 创建一个 ModelCheckpoint 回调函数来保存最佳模型
checkpoint = ModelCheckpoint('lstm_model_20.keras', monitor='val_loss', save_best_only=True, mode='min')

# 训练模型
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0,
          callbacks=[history, lr_reducer, early_stopping, checkpoint])

# 打印出最佳迭代轮数的索引和对应的验证集损失
best_epoch = np.argmin(history.metrics['val_loss'])
print(f"Best epoch with minimum validation loss: {best_epoch}")
print(f"Minimum validation loss achieved: {history.metrics['val_loss'][best_epoch]}")

# 绘制训练和验证损失曲线
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(range(len(history.metrics['loss'])), history.metrics['loss'], 'g-')
ax1.plot(range(len(history.metrics['val_loss'])), history.metrics['val_loss'], 'r--')
ax2.plot(range(len(history.metrics['mae'])), history.metrics['mae'], 'b-')
ax2.plot(range(len(history.metrics['val_mae'])), history.metrics['val_mae'], 'orange--')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='g')
ax2.set_ylabel('MAE', color='b')
plt.title('Training and Validation Losses')
plt.show()
'''


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.title('model train loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper right')
plt.show()



# 加载模型用于预测
model = tensorflow.keras.models.load_model('lstm_model_25.h5')

# 进行预测
predicted_soc = model.predict(X_test)

# 反归一化以获得真实的SOC值
#predicted_soc = predicted_soc*(scaler_labels_max-scaler_labels_min)+scaler_labels_min
#y_test = y_test*(scaler_labels_max-scaler_labels_min)+scaler_labels_min

#predicted_soc = predicted_soc*scaler_labels_std+scaler_labels_mean
#y_test = y_test*scaler_labels_std+scaler_labels_mean

predicted_soc=predicted_soc*100
y_test=y_test*100

# 打印一些结果
#print("Predicted SOC:", predicted_soc[0:10,])
#print("True SOC:", y_test[0:10,])


# 模型评估
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score

print('**************************输出测试集的模型评估指标结果*******************************')

R=round(r2_score(y_test, predicted_soc), 4)
print('R^2：', round(r2_score(y_test, predicted_soc), 4))
print('均方误差:', round(mean_squared_error(y_test, predicted_soc), 4))
print('解释方差分:', round(explained_variance_score(y_test, predicted_soc), 4))
print('绝对误差:', round(mean_absolute_error(y_test, predicted_soc), 4))



def mean_absolute_error(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mape = np.mean(np.abs(actual- predicted))
    return mape

# 计算 MAPE
mape_value = mean_absolute_error(y_test[1:], predicted_soc[1:,0])
print(f"平均绝对误差: {mape_value:.2f}%")


# 计算误差
errors = predicted_soc[1:,0]-y_test[1:]


#file_path = 'output.csv'
#pd.DataFrame(predicted_soc).to_csv(file_path, index=False)
#print(f"数组已成功保存到 {file_path}")

plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.plot(range(len(y_test)), y_test, color="blue", linewidth=1.5, linestyle="-")
plt.plot(range(len(predicted_soc)), predicted_soc , color="red", linewidth=1.5, linestyle="-.")
plt.legend(['真实值', '预测值'])
#plt.title(f"LSTM回归模型真实值与预测值比对图")
plt.title(f"LSTM回归模型真实值与预测值比对图,平均误差:{mape_value:.2f}%")
plt.show()


# 绘制误差分布图
plt.figure()
plt.hist(errors, bins=20, edgecolor='black')
plt.title('Error Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.show()

# 绘制真实值 vs 预测值的散点图
plt.figure()
plt.scatter(y_test, predicted_soc,alpha=0.5)
plt.title(f"True Values vs Predicted Values,R^2:{R:.2f}")
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)  # 理想情况下通过原点的直线
plt.title(f"True Values vs Predicted Values,R^2:{R:.2f}")
plt.show()

