import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error as rmse
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score


def soc_handle(df):
    result = pd.DataFrame({
        "soc_bms_err_75_80": [],
        "soc_lstm_err_75_80": [],
        "soc_bms_err_80_85": [],
        "soc_lstm_err_80_85": [],
        "soc_lstm_err_85_90": [],
        "soc_bms_err_85_90": [],
        "soc_bms_err_90_95": [],
        "soc_lstm_err_90_95": [],
        "soc_bms_err_95_100": [],
        "soc_lstm_err_95_100": []
    })
    flgVld_75_80 = (df['TrueSoc'] < 80) & (df['TrueSoc'] >= 75)
    flgVld_80_85 = (df['TrueSoc'] < 85) & (df['TrueSoc'] >= 80)
    flgVld_85_90 = (df['TrueSoc'] < 90) & (df['TrueSoc'] >= 85)
    flgVld_90_95 = (df['TrueSoc'] < 95) & (df['TrueSoc'] >= 90)
    flgVld_95_100 = (df['TrueSoc'] <= 100) & (df['TrueSoc'] >= 95)

    soc_bms_err_75_80 = df['BmsSoc'][flgVld_75_80] - df['TrueSoc'][flgVld_75_80]
    soc_lstm_err_75_80 = df['preSoc'][flgVld_75_80] - df['TrueSoc'][flgVld_75_80]

    soc_bms_err_80_85 = df['BmsSoc'][flgVld_80_85] - df['TrueSoc'][flgVld_80_85]
    soc_lstm_err_80_85 = df['preSoc'][flgVld_80_85] - df['TrueSoc'][flgVld_80_85]

    soc_bms_err_85_90 = df['BmsSoc'][flgVld_85_90] - df['TrueSoc'][flgVld_85_90]
    soc_lstm_err_85_90 = df['preSoc'][flgVld_85_90] - df['TrueSoc'][flgVld_85_90]

    soc_bms_err_90_95 = df['BmsSoc'][flgVld_90_95] - df['TrueSoc'][flgVld_90_95]
    soc_lstm_err_90_95 = df['preSoc'][flgVld_90_95] - df['TrueSoc'][flgVld_90_95]

    soc_bms_err_95_100 = df['BmsSoc'][flgVld_95_100] - df['TrueSoc'][flgVld_95_100]
    soc_lstm_err_95_100 = df['preSoc'][flgVld_95_100] - df['TrueSoc'][flgVld_95_100]

    newresult = pd.DataFrame({
        "soc_bms_err_75_80":soc_bms_err_75_80 ,
        "soc_lstm_err_75_80":soc_lstm_err_75_80,
        "soc_bms_err_80_85": soc_bms_err_80_85,
        "soc_lstm_err_80_85": soc_lstm_err_80_85,
        "soc_bms_err_85_90": soc_bms_err_85_90,
        "soc_lstm_err_85_90": soc_lstm_err_85_90,
        "soc_bms_err_90_95": soc_bms_err_90_95,
        "soc_lstm_err_90_95": soc_lstm_err_90_95,
        "soc_bms_err_95_100": soc_bms_err_95_100,
        "soc_lstm_err_95_100": soc_lstm_err_95_100
    })
    result = pd.concat([result, newresult])
    return result



df= pd.read_csv('output_data.csv')
mf=soc_handle(df)
# 将DataFrame写入CSV文件
#mf.to_csv('output_plot.csv', index=False)


# 计算均值
mean_value_soc_lstm_err_75_80 = np.mean(mf['soc_lstm_err_75_80'])
mean_value_soc_lstm_err_80_85 = np.mean(mf['soc_lstm_err_80_85'])
mean_value_soc_lstm_err_85_90 = np.mean(mf['soc_lstm_err_85_90'])
mean_value_soc_lstm_err_90_95 = np.mean(mf['soc_lstm_err_90_95'])
mean_value_soc_lstm_err_95_100 = np.mean(mf['soc_lstm_err_95_100'])
# 计算方差
variance_value_soc_lstm_err_75_80 = np.var(mf['soc_lstm_err_75_80'])
variance_value_soc_lstm_err_80_85 = np.var(mf['soc_lstm_err_80_85'])
variance_value_soc_lstm_err_85_90 = np.var(mf['soc_lstm_err_85_90'])
variance_value_soc_lstm_err_90_95 = np.var(mf['soc_lstm_err_90_95'])
variance_value_soc_lstm_err_95_100 = np.var(mf['soc_lstm_err_95_100'])

#计算最大误差:
max_value_soc_lstm_err_75_80 = np.max(np.abs(mf['soc_lstm_err_75_80']))
max_value_soc_lstm_err_80_85 = np.max(np.abs(mf['soc_lstm_err_80_85']))
max_value_soc_lstm_err_85_90 = np.max(np.abs(mf['soc_lstm_err_85_90']))
max_value_soc_lstm_err_90_95 = np.max(np.abs(mf['soc_lstm_err_90_95']))
max_value_soc_lstm_err_95_100 = np.max(np.abs(mf['soc_lstm_err_95_100']))

#计算均方根误差:
rmse_soc_lstm_err_75_80=np.sqrt(variance_value_soc_lstm_err_75_80)
rmse_soc_lstm_err_80_85=np.sqrt(variance_value_soc_lstm_err_80_85)
rmse_soc_lstm_err_85_90=np.sqrt(variance_value_soc_lstm_err_85_90)
rmse_soc_lstm_err_90_95=np.sqrt(variance_value_soc_lstm_err_90_95)
rmse_soc_lstm_err_95_100=np.sqrt(variance_value_soc_lstm_err_95_100)

print('max_value_soc_lstm_err_75_80:',max_value_soc_lstm_err_75_80)
print('max_value_soc_lstm_err_80_85:',max_value_soc_lstm_err_80_85)
print('max_value_soc_lstm_err_85_90:',max_value_soc_lstm_err_85_90)
print('max_value_soc_lstm_err_90_95:',max_value_soc_lstm_err_90_95)
print('max_value_soc_lstm_err_95_100:',max_value_soc_lstm_err_95_100)


print('rmse_value_soc_lstm_err_75_80:',rmse_soc_lstm_err_75_80)
print('rmse_value_soc_lstm_err_80_85:',rmse_soc_lstm_err_80_85)
print('rmse_value_soc_lstm_err_85_90:',rmse_soc_lstm_err_85_90)
print('rmse_value_soc_lstm_err_90_95:',rmse_soc_lstm_err_90_95)
print('rmse_value_soc_lstm_err_95_100:',rmse_soc_lstm_err_95_100)
# 绘制频率图
plt.figure()
plt.hist(mf['soc_lstm_err_75_80'].values, bins=20,edgecolor='black')
plt.axvline(mean_value_soc_lstm_err_75_80, color='red', linestyle='dashed', linewidth=1, label='Mean')
plt.axvline(mean_value_soc_lstm_err_75_80 + np.sqrt(variance_value_soc_lstm_err_75_80), color='blue', linestyle='dashed', linewidth=1, label='Mean ± Std Dev')
plt.axvline(mean_value_soc_lstm_err_75_80 - np.sqrt(variance_value_soc_lstm_err_75_80), color='blue', linestyle='dashed', linewidth=1)
plt.title('Frequency Histogram of soc_lstm_err_75_80')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 绘制频率图
plt.figure()
plt.hist(mf['soc_lstm_err_80_85'], bins=20, edgecolor='black')
plt.axvline(mean_value_soc_lstm_err_80_85, color='red', linestyle='dashed', linewidth=1, label='Mean')
plt.axvline(mean_value_soc_lstm_err_80_85 + np.sqrt(variance_value_soc_lstm_err_80_85), color='blue', linestyle='dashed', linewidth=1, label='Mean ± Std Dev')
plt.axvline(mean_value_soc_lstm_err_80_85 - np.sqrt(variance_value_soc_lstm_err_80_85), color='blue', linestyle='dashed', linewidth=1)
plt.title('Frequency Histogram of soc_lstm_err_80_85')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 绘制频率图
plt.figure()
plt.hist(mf['soc_lstm_err_85_90'], bins=20, edgecolor='black')
plt.axvline(mean_value_soc_lstm_err_85_90, color='red', linestyle='dashed', linewidth=1, label='Mean')
plt.axvline(mean_value_soc_lstm_err_85_90 + np.sqrt(variance_value_soc_lstm_err_85_90), color='blue', linestyle='dashed', linewidth=1, label='Mean ± Std Dev')
plt.axvline(mean_value_soc_lstm_err_85_90 - np.sqrt(variance_value_soc_lstm_err_85_90), color='blue', linestyle='dashed', linewidth=1)
plt.title('Frequency Histogram of soc_lstm_err_85_90')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 绘制频率图
plt.figure()
plt.hist(mf['soc_lstm_err_90_95'], bins=20, edgecolor='black')
plt.axvline(mean_value_soc_lstm_err_90_95, color='red', linestyle='dashed', linewidth=1, label='Mean')
plt.axvline(mean_value_soc_lstm_err_90_95 + np.sqrt(variance_value_soc_lstm_err_90_95), color='blue', linestyle='dashed', linewidth=1, label='Mean ± Std Dev')
plt.axvline(mean_value_soc_lstm_err_90_95 - np.sqrt(variance_value_soc_lstm_err_90_95), color='blue', linestyle='dashed', linewidth=1)
plt.title('Frequency Histogram of soc_lstm_err_90_95')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 绘制频率图
plt.figure()
plt.hist(mf['soc_lstm_err_95_100'], bins=20, edgecolor='black')
plt.axvline(mean_value_soc_lstm_err_95_100, color='red', linestyle='dashed', linewidth=1, label='Mean')
plt.axvline(mean_value_soc_lstm_err_95_100 + np.sqrt(variance_value_soc_lstm_err_95_100), color='blue', linestyle='dashed', linewidth=1, label='Mean ± Std Dev')
plt.axvline(mean_value_soc_lstm_err_95_100 - np.sqrt(variance_value_soc_lstm_err_95_100), color='blue', linestyle='dashed', linewidth=1)
plt.title('Frequency Histogram of soc_lstm_err_95_100')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()








