# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy.linalg as la
import math
from sklearn.svm import SVR
#from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

###### baseline for one step prediction  ############
train_rate = 0.8           # 80% of data is used for training
seq_len = 3                # input sequence length
pre_len = 1                # output sequence length
dataset='Rumor'          # choose from {'SH_Park','CN_AQI','Metr-LA','PeMS08','COVID',‘Rumor’}
method = 'SVR'              # LM or HA or SVR or ARIMA

np.seterr(divide = 'ignore')


def preprocess_data(data, time_len, rate, seq_len, pre_len, norm=True):
    data1 = np.mat(data)
    
    max_num=np.max(data1)
    min_num=np.min(data1)   
    # Norm
    if max_num!=min_num and norm==True:
        data1=(data1-max_num)/(max_num-min_num)

    train_size = int(time_len * rate)
    train_data = data1[0:train_size]
    test_data = data1[train_size:time_len]  # split by time
    
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]   # sliding windows
        trainX.append(a[0 : seq_len])
        trainY.append(a[seq_len : seq_len + pre_len])
    for i in range(len(test_data) - seq_len -pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0 : seq_len])
        testY.append(b[seq_len : seq_len + pre_len])
    return trainX, trainY, testX, testY, max_num, min_num
    
###### evaluation ######
def evaluation(a,b,max_num, min_num):
    # Recover
    a=a*(max_num-min_num)+min_num
    b=b*(max_num-min_num)+min_num

    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b)/la.norm(a)
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a - b))/np.var(a)
    if np.var(a) == 0:
        r2 = var = 0  # for ARIMA
    return rmse, mae, 1-F_norm, r2, var
 
if dataset=='SH_Park':
    path='../data/SH_Park/SH_Park_10_d.csv'
elif dataset=='CN_AQI':
    path='../data/CN_AQI/AQI_data.csv'
elif dataset=='Metr-LA':
    path='../data/Metr-LA/Metr_LA.csv'
elif dataset=='PeMS08':
    path='../data/PeMS08/PeMS08_Flow.csv' ### to be continued...
elif dataset=='COVID':
    path='../data/COVID/covid_us.csv' 
elif dataset=='Rumor':
    path='../data/SimRumor/Rumor_S.csv'

data = pd.read_csv(path)

time_len = data.shape[0]   # number of time slots
num_nodes = data.shape[1]  # node number
trainX,trainY,testX,testY, max_num, min_num= preprocess_data(data, time_len, train_rate, seq_len, pre_len)



########### LM #############
if method == 'LM':
    result = []
    for i in range(len(testX)):
        a = np.array(testX[i])   # draw one sample
        tempResult = []

        a1=a[-1]
        tempResult.append(a1)

        result.append(tempResult)
    result1 = np.array(result)
    result1 = np.reshape(result1, [-1,num_nodes])
    testY1 = np.array(testY)
    testY1 = np.reshape(testY1, [-1,num_nodes])
    rmse, mae, accuracy,r2,var = evaluation(testY1, result1, max_num, min_num)
    print('LM_rmse:%r'%rmse,
          'LM_mae:%r'%mae,
          'LM_acc:%r'%accuracy,
          'LM_r2:%r'%r2,
          'LM_var:%r'%var)



########### HA #############
if method == 'HA':
    result = []
    for i in range(len(testX)):
        a = np.array(testX[i])   # draw one sample
        tempResult = []

        a1 = np.mean(a, axis=0)
        tempResult.append(a1)


        result.append(tempResult)
    result1 = np.array(result)
    result1 = np.reshape(result1, [-1,num_nodes])
    testY1 = np.array(testY)
    testY1 = np.reshape(testY1, [-1,num_nodes])
    rmse, mae, accuracy,r2,var = evaluation(testY1, result1, max_num, min_num)
    print('HA_rmse:%r'%rmse,
          'HA_mae:%r'%mae,
          'HA_acc:%r'%accuracy,
          'HA_r2:%r'%r2,
          'HA_var:%r'%var)


############ SVR #############
if method == 'SVR':  
    total_rmse, total_mae, total_acc, result = [], [],[],[]
    for i in range(num_nodes):
        print('Node: '+str(i))

        data1 = np.mat(data)
        max_num=np.max(data1)
        min_num=np.min(data1)

        if np.max(data1)!=np.min(data1):
            data1=(data1-np.max(data1))/(np.max(data1)-np.min(data1)) # norm in advance

        a = data1[:,i]     # predict each node separately
        a_X, a_Y, t_X, t_Y, _, _= preprocess_data(a, time_len, train_rate, seq_len, pre_len, norm=False)
        a_X = np.array(a_X)
        a_X = np.reshape(a_X,[-1, seq_len])
        a_Y = np.array(a_Y)
        a_Y = np.reshape(a_Y,[-1, pre_len])
        a_Y = np.mean(a_Y, axis=1)
        t_X = np.array(t_X)
        t_X = np.reshape(t_X,[-1, seq_len])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y,[-1, pre_len])    
       
        svr_model=SVR(kernel='linear')
        svr_model.fit(a_X, a_Y)
        pre = svr_model.predict(t_X)
        result.append(pre)
    

    result1 = np.array(result)
    result1 = np.reshape(result1, [num_nodes,-1])
    result1 = np.transpose(result1)
    testY1 = np.array(testY)


    testY1 = np.reshape(testY1, [-1,num_nodes])
    total = np.mat(total_acc)
    total[total<0] = 0
    rmse1, mae1, acc1,r2,var = evaluation(testY1, result1, max_num, min_num)
    print('SVR_rmse:%r'%rmse1,
          'SVR_mae:%r'%mae1,
          'SVR_acc:%r'%acc1,
          'SVR_r2:%r'%r2,
          'SVR_var:%r'%var)
    


# ######## ARIMA #########
# if method == 'ARIMA':
#     rng = pd.date_range('12/1/2021', periods=time_len, freq=str(delta_t)+'min')
#     a1 = pd.DatetimeIndex(rng)
#     data.index = a1
#     num = data.shape[1]   
#     rmse,mae,acc,r2,var,pred,ori = [],[],[],[],[],[],[]
#     for i in range(num_nodes):
#         ts = data.iloc[:,i]
#         ts_log=np.log(ts)    
#         ts_log=np.array(ts_log,dtype=float)
#         where_are_inf = np.isinf(ts_log)
#         ts_log[where_are_inf] = 0
#         ts_log = pd.Series(ts_log)
#         ts_log.index = a1
#         model = sm.tsa.arima.ARIMA(ts_log,order=[1,0,0])
#         properModel = model.fit()
#         predict_ts = properModel.predict(4, dynamic=True)
#         log_recover = np.exp(predict_ts)
#         ts = ts[log_recover.index]
#         print(ts.shape,log_recover.shape)
#         er_rmse,er_mae,er_acc,r2_score,var_score = evaluation(ts,log_recover)
#         rmse.append(er_rmse)
#         mae.append(er_mae)
#         acc.append(er_acc)
#         r2.append(r2_score)
#         var.append(var_score)
# #    for i in range(109,num):
# #        ts = data.iloc[:,i]
# #        ts_log=np.log(ts)    
# #        ts_log=np.array(ts_log,dtype=np.float)
# #        where_are_inf = np.isinf(ts_log)
# #        ts_log[where_are_inf] = 0
# #        ts_log = pd.Series(ts_log)
# #        ts_log.index = a1
# #        model = ARIMA(ts_log,order=[1,1,1])
# #        properModel = model.fit(disp=-1, method='css')
# #        predict_ts = properModel.predict(2, dynamic=True)
# #        log_recover = np.exp(predict_ts)
# #        ts = ts[log_recover.index]
# #        er_rmse,er_mae,er_acc,r2_score,var_score = evaluation(ts,log_recover)
# #        rmse.append(er_rmse)
# #        mae.append(er_mae)
# #        acc.append(er_acc)  
# #        r2.append(r2_score)
# #        var.append(var_score)
#     acc1 = np.mat(acc)
#     acc1[acc1 < 0] = 0
#     print('arima_rmse:%r'%(np.mean(rmse)),
#           'arima_mae:%r'%(np.mean(mae)),
#           'arima_acc:%r'%(np.mean(acc1)),
#           'arima_r2:%r'%(np.mean(r2)),
#           'arima_var:%r'%(np.mean(var)))
  