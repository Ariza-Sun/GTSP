# calculate the autocorrelation function for SH_park.csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def autocorrelation_function(x):
    n = len(x)
    mean = np.mean(x)
    acf_values = []


    for k in range(n):
        if np.sum((x - mean)**2)!=0:
            acf_k = np.sum((x[:n-k] - mean) * (x[k:] - mean)) / (np.sum((x - mean)**2))
        else:
            acf_k=0
        acf_values.append(acf_k)
    
    return np.array(acf_values)


# ReadFile
data=pd.read_csv('../data/SH_Park_10.csv').values
N=data.shape[1]
acf_list=[]


# 计算自相关函数
for m in range(N):
    acf_list.append(autocorrelation_function(data[0:40,m]))

acf_list=np.array(acf_list)
acf_values=np.mean(acf_list,axis=0)



# 绘制自相关函数图像
#plt.stem(range(len(acf_values)), acf_values, basefmt="b-", use_line_collection=True)

markerline, stemline, baseline,=plt.stem(range(len(acf_values)), acf_values,basefmt="b-", linefmt='b-', markerfmt='ro', use_line_collection=True)  # 大小可调
plt.setp(stemline, linewidth = 0.25)
plt.setp(markerline, markersize = 0.5)


plt.title('Autocorrelation Function')
plt.xlabel('Lag K')
plt.ylabel('Autocorrelation')
plt.savefig('ACF.png')

