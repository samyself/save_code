import pandas as pd
import numpy as np
df= pd.read_csv('output_data.csv')
m=len(df['errors'])
xs=[]
for i in range(0,m,1):
    if (np.abs(df['errors'][i]) >5) & (df['TrueSoc'][i]>=80):
        x=df['chargetimes'][i]
        xs.append(x)
unique_arr=[]
for element in xs:
    if element not in unique_arr:
       unique_arr.append(element)
print(len(unique_arr))

file_path = 'output_error5_chargetimes.csv'
pd.DataFrame(unique_arr).to_csv(file_path, index=False)
print(f"数组已成功保存到 {file_path}")


