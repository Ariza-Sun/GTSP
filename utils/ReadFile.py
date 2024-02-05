import torch
import numpy as np
import pandas as pd
import os
import math
import pickle
#from torch.utils.data.dataset import TensorDataset
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import roc_curve, auc
#from sgm import sgraphmatch
import torch.nn.functional as F
import networkx as nx
from networkx.convert import from_dict_of_dicts
from networkx.classes.graph import Graph

# if use cuda
use_cuda = torch.cuda.is_available()


class MyDataset(Dataset):
    # 构造函数
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    # 返回数据集大小
    def __len__(self):
        return self.data_tensor.size(0)
    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]


def load_parking(batch_size=128,network='ER',dataset='SH_Park'):

    if dataset=='SH_Park':
        path='data/SH_Park/SH_Park_10_d.csv'
    elif dataset=='CN_AQI':
        path='data/CN_AQI/AQI_data.csv'
    elif dataset=='Metr-LA':
        path='data/Metr-LA/Metr_LA.csv'
    elif dataset=='PeMS08':
        path='data/PeMS08/PeMS08_Flow.csv' 
    elif dataset=='COVID':
        path='data/COVID/covid_us.csv' 



    #edges=pd.read_csv().values
    edges=np.zeros((1,1))  # None
    loc=pd.read_csv(path).values
    if np.max(loc)!=np.min(loc): #norm
        loc=(loc-np.max(loc))/(np.max(loc)-np.mean(loc))

    loc=torch.from_numpy(loc)  #(time, node)
    node_num=loc.shape[1]
    loc = torch.unsqueeze(loc, dim=-1)
     
    print('Scene:'+dataset)
    print("Network: "+network)
    print("Node Number: "+str(node_num))

    P = 2
    sample = int(loc.size(0)/P)
    node = loc.size(1)
    dim = 1

    data = torch.zeros(sample,node,P,dim)  
    for i in range(data.size(0)):
        data[i] = loc[i*P:(i+1)*P].transpose(0,1) # turn it into pairs

    # cut to train val and test
    train_data = data[:int(sample*5/7)]  #（sample,node,P,dim）
    val_data = data[int(sample*5/7):int(sample*6/7)]
    test_data = data[int(sample*6/7):]

    # comment: fail to use the right form of dataloader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader,test_loader,torch.from_numpy(edges),node_num


def load_gru_parking(batchsize,seq_len,pre_len,dataset,norm=True):
    
    if dataset=='SH_Park':
        path='data/SH_Park/SH_Park_10_d.csv'
    elif dataset=='CN_AQI':
        path='data/CN_AQI/AQI_data.csv'
    elif dataset=='Metr-LA':
        path='data/Metr-LA/Metr_LA.csv'
    elif dataset=='PeMS08':
        path='data/PeMS08/PeMS08_Flow.csv' 
    elif dataset=='COVID':
        path='data/COVID/covid_us.csv' 



    raw_data=pd.read_csv(path).values
    node_num=raw_data.shape[1]

        # Norm
    if np.max(raw_data)!=np.min(raw_data) and norm==True:
        raw_data=(raw_data-np.max(raw_data))/(np.max(raw_data)-np.mean(raw_data))


    time_slot=raw_data.shape[0]
    train_size=int(time_slot*0.7)
    # train_raw=raw_data[:train_size]
    # test_raw=raw_data[train_size:]

    # create dataset by sliding windows
    train_x,train_y,test_x,test_y=[],[],[],[]
    for iter in range(train_size-seq_len-pre_len+1):
        train_x.append(raw_data[iter:iter+seq_len,:])
        train_y.append(raw_data[iter+seq_len:iter+seq_len+pre_len,:])

    for iter in range(train_size,time_slot-seq_len-pre_len+1):
        test_x.append(raw_data[iter:iter+seq_len,:])
        test_y.append(raw_data[iter+seq_len:iter+seq_len+pre_len,:])

    train_x=torch.from_numpy(np.array(train_x,dtype=float))
    train_y=torch.from_numpy(np.array(train_y,dtype=float))
    test_x=torch.from_numpy(np.array(test_x,dtype=float))
    test_y=torch.from_numpy(np.array(test_y,dtype=float))

    # convert dtype
    train_x=train_x.double()
    train_y=train_y.double()
    test_x=test_x.double()
    test_y=test_y.double()


    train_set=MyDataset(train_x,train_y)
    test_set=MyDataset(test_x,test_y)  # torch.utils.data.random_split
    

    train_loader=DataLoader(train_set,batch_size=batchsize, shuffle=True)
    test_loader=DataLoader(test_set,batch_size=batchsize,shuffle=True)

    return train_loader,test_loader,node_num
    


def load_TS_parking(batch_size=128,node_num=148,network='ER',delta_t=10,seq_len=3,pre_len=1):   
    # Specially designed for RAIDD
    # Normalized=True

    raw_data=pd.read_csv("../data/SH_Park_"+str(delta_t)+".csv").values #(time_slots,node_num)
    min_val=np.min(raw_data)
    max_val=np.max(raw_data)

    raw_data=(raw_data-min_val)/(max_val-min_val)


    time_slot=raw_data.shape[0]
    train_size=int(time_slot*0.7)
    # train_raw=raw_data[:train_size]
    # test_raw=raw_data[train_size:]

    # create dataset by sliding windows
    train_x,train_y,test_x,test_y=[],[],[],[]
    for iter in range(train_size-seq_len-pre_len+1):
        train_x.append(raw_data[iter:iter+seq_len,:])               
        train_y.append(raw_data[iter+seq_len:iter+seq_len+pre_len,:])

    for iter in range(train_size,time_slot-seq_len-pre_len+1):
        test_x.append(raw_data[iter:iter+seq_len,:])
        test_y.append(raw_data[iter+seq_len:iter+seq_len+pre_len,:])

    train_x=torch.from_numpy(np.array(train_x,dtype=float))
    train_y=torch.from_numpy(np.array(train_y,dtype=float))
    test_x=torch.from_numpy(np.array(test_x,dtype=float))
    test_y=torch.from_numpy(np.array(test_y,dtype=float))

    # convert dtype
    train_x=train_x.double() #(,seq_len,node_num)
    train_y=train_y.double() #(,pre_len,node_num)
    test_x=test_x.double()
    test_y=test_y.double()


    train_set=MyDataset(train_x,train_y)
    test_set=MyDataset(test_x,test_y)  # torch.utils.data.random_split
    

    train_loader=DataLoader(train_set,batch_size=batch_size, shuffle=True)
    test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=True)


    edges=pd.read_csv("../data/connect_SH_adj.csv",header=None).values
    return train_loader,test_loader,torch.from_numpy(edges),min_val,max_val




def load_TS_parking_ATGCN(batch_size=128,network='ER',seq_len=3,pre_len=1,dataset='SH_Park'): 
    # Specially designed for ATGCN
    # Normalized=True

    if dataset=='SH_Park':
        path='../data/SH_Park/SH_Park_10_d.csv'
    elif dataset=='CN_AQI':
        path='../data/CN_AQI/AQI_data.csv'
    elif dataset=='Metr-LA':
        path='../data/Metr-LA/Metr_LA.csv'
    elif dataset=='PeMS08':
        path='../data/PeMS08/PeMS08_Flow.csv' 
    elif dataset=='COVID':
        path='../data/COVID/covid_us.csv' 

    raw_data=pd.read_csv(path).values #(time_slots,node_num)
    min_val=np.min(raw_data)
    max_val=np.max(raw_data)

    raw_data=(raw_data-min_val)/(max_val-min_val)


    time_slot=raw_data.shape[0]
    train_size=int(time_slot*0.7)
    node_num=raw_data.shape[1]
    # train_raw=raw_data[:train_size]
    # test_raw=raw_data[train_size:]
    print('Scene:'+dataset)
    print("Network: "+network)
    print("Node Number: "+str(node_num))

    # create dataset by sliding windows
    train_x,train_y,test_x,test_y=[],[],[],[]
    for iter in range(train_size-seq_len-pre_len+1):
        train_x.append(raw_data[iter:iter+seq_len,:])               
        train_y.append(raw_data[iter+seq_len:iter+seq_len+pre_len,:])

    for iter in range(train_size,time_slot-seq_len-pre_len+1):
        test_x.append(raw_data[iter:iter+seq_len,:])
        test_y.append(raw_data[iter+seq_len:iter+seq_len+pre_len,:])

    train_x=torch.from_numpy(np.array(train_x,dtype=float))
    train_y=torch.from_numpy(np.array(train_y,dtype=float))
    test_x=torch.from_numpy(np.array(test_x,dtype=float))
    test_y=torch.from_numpy(np.array(test_y,dtype=float))

    # convert dtype
    train_x=train_x.type(torch.float32) #(,seq_len,node_num)
    train_y=train_y.type(torch.float32)#(,pre_len,node_num)
    test_x=test_x.type(torch.float32)
    test_y=test_y.type(torch.float32)


    train_set=MyDataset(train_x,train_y)
    test_set=MyDataset(test_x,test_y)  # torch.utils.data.random_split
    

    train_loader=DataLoader(train_set,batch_size=batch_size, shuffle=True)
    test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=True)


    edges=np.zeros((1,1))  # None
    return train_loader,test_loader,torch.from_numpy(edges),node_num




