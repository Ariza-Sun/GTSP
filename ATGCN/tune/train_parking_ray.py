import torch
import time
import torch.nn.utils as U
import torch.optim as optim
import ray
from ray import tune
from ray import train
import argparse
import numpy as np


import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import torchmetrics


import pandas as pd
import os
import math
from sklearn.metrics import roc_curve, auc


import scipy.sparse as sp
from torch.nn.functional import normalize


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from functools import partial


import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# from model.ATGCN import *
#from utils.tools import *
# from utils.ReadFile import* 
# from utils.eval import *

#from torch.utils.tensorboard import SummaryWriter   


# config designed for SH_Park.csv
HYP = {
    'node_size': 148,
    'hid': 128,  # hidden unit size
    'dim':1,
    'epoch_num': 1500,  # epoch
    'batch_size': 128,  # batch size
    'lr_net': 0.004,  # lr for net generator 0.004   # network structure inference  ####revised
    'lr_dyn': 0.001,  # lr for dyn learner           # dynamic reconstruction
    'lr_stru': 0.0001,  # lr for structural loss 0.0001 2000 0.01  0.00001
    'hard_sample': False,  # weather to use hard mode in gumbel
    'sample_time': 1,  # sample time while training
    'temp': 1,  # temperature
    'drop_frac': 1,  # temperature drop frac
    'seq_len':3,            
    'pre_len':1,
    'delta_t':10   # Time Slice
}


parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=148, help='Number of nodes, default=148')
parser.add_argument('--network', type=str, default='ER', help='type of network')
parser.add_argument('--sys', type=str, default='parking', help='simulated system to model,spring or cmn')
parser.add_argument('--dim', type=int, default=1, help='# information dimension of each node parking:1 ')
parser.add_argument('--delta_t', type=int, default=10, help='time_interval, default=15min')
parser.add_argument('--device_id', type=int, default=1, help='Gpu_id, default=1')
args = parser.parse_args()

#set gpu id
#print(torch.cuda.is_available())
torch.cuda.set_device(args.device_id)
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



###############  model ###############

def calculate_laplacian_with_self_loop(matrix):
    eye=torch.eye(matrix.size(0)).to(device)
    matrix=matrix.to(device)
    matrix = matrix + eye
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )

    return normalized_laplacian
class TGCNGraphConvolution(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units  #64
        self._output_dim = output_dim
        self._bias_init_value = bias
        # self.register_buffer(
        #     "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        # )
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state, adj_mat):
        
        # Deal with dynamic graph
        laplacian= calculate_laplacian_with_self_loop(adj_mat)

        batch_size, num_nodes = inputs.shape   # warning inputs.shape[0]

        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
  
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
  
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + 1) * batch_size)
        )
        
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)

        a_times_concat = torch.matmul(laplacian,concatenation)  

        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + 1, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)

        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TGCNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        #self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
             self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state, adj_mat):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state, adj_mat)) # warning concate=nan
  
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state, adj_mat))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
  
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class ATGCN(nn.Module):   # Wrapper
    def __init__(self, adj, hidden_dim: int, output_dim):
        super(ATGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self._output_dim=output_dim
        #self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self._input_dim, self._hidden_dim)
        self.Linear=nn.Linear(self._hidden_dim,self._output_dim)

    def forward(self, inputs, adj_mat): # inputs:(batch_size, seq_len, node_num)    
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        output = None

        for i in range(seq_len):  # Through Mutiple cells
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state, adj_mat)   #(output=hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        output=self.Linear(output)
        output = output.reshape((batch_size, self._output_dim, num_nodes))
        return output


class Gumbel_Generator_Old(nn.Module):  
    def __init__(self, sz=10, temp=10, temp_drop_frac=0.9999):
        super(Gumbel_Generator_Old, self).__init__()
        self.sz = sz  # N
        self.gen_matrix = Parameter(torch.rand(sz, sz, 2)) 
        self.temperature = temp
        self.temp_drop_frac = temp_drop_frac

    def drop_temp(self):
        # drop temperature
        self.temperature = self.temperature * self.temp_drop_frac

    # output: a matrix
    def sample_all(self, hard=False):
        self.logp = self.gen_matrix.view(-1, 2) 
        out = gumbel_softmax(self.logp, self.temperature, hard) # different shape for hard seems
        if hard:
            hh = torch.zeros(self.gen_matrix.size()[0] ** 2, 2)
            for i in range(out.size()[0]):  # N
                hh[i, out[i]] = 1
            out = hh
        if use_cuda:
            out = out.cuda()
        out_matrix = out[:, 0].view(self.gen_matrix.size()[0], self.gen_matrix.size()[0]) # (N,N)
 

        return out_matrix

    # output: the i-th column of matrix
    def sample_adj_i(self, i, hard=False, sample_time=1):
        self.logp = self.gen_matrix[:, i] #（N，2）
        out = gumbel_softmax(self.logp, self.temperature, hard=hard) # (N,2)/(N,1)
        if use_cuda:
            out = out.cuda()
        if hard:
            out_matrix = out.float()
        else:
            out_matrix = out[:, 0] 
        return out_matrix    # (N,1)

    def get_temperature(self):
        return self.temperature

    def init(self, mean, var):
        init.normal_(self.gen_matrix, mean=mean, std=var)


class Controller(nn.Module):


    def __init__(self, dim, hid):
        super(Controller, self).__init__()
        self.dim = dim

        self.hid = hid
        self.linear1 = nn.Linear(self.dim,self.hid)
        self.linear2 = nn.Linear(self.hid,self.hid)
        self.linear3 = nn.Linear(self.hid, self.hid)
        self.linear4 = nn.Linear(self.hid, self.hid)
        self.output = nn.Linear(self.hid+self.dim,self.dim//2)

    def forward(self, cur_x,diff_x,adj_col,except_control_node_list):

        starter = diff_x  # 128,10,4

        x = F.relu(self.linear1(starter))  # 128,10,256


        x = F.relu(self.linear2(x))  # 128,10,256
        adj_col = adj_col[except_control_node_list]


        x = x * adj_col.unsqueeze(1).expand(adj_col.size(0), self.hid)  # 128,10,256

        x_sum = torch.sum(x, 1)  # 128,256


        x = F.relu(self.linear3(x_sum))  # 128,256
        x = F.relu(self.linear4(x))  # 128,256

        x = torch.cat((cur_x, x), dim=-1)  # 128,256+4
        x  = self.output(x)  # 128,4
        #x = torch.sigmoid(x) # if dyn is CMN

        return x


'''network completetion'''

class Gumbel_Generator_nc(nn.Module):
    def __init__(self, sz=10, del_num=1, temp=10, temp_drop_frac=0.9999):
        super(Gumbel_Generator_nc, self).__init__()
        self.sz = sz
        self.del_num = del_num
        self.gen_matrix = Parameter(
            torch.rand(del_num * (2 * sz - del_num - 1) // 2, 2))  # cmy get only unknown part parameter
        self.temperature = temp
        self.temp_drop_frac = temp_drop_frac

    def drop_temp(self):
        # 降温过程
        self.temperature = self.temperature * self.temp_drop_frac

    def sample_all(self, hard=False):
        self.logp = self.gen_matrix
        if use_cuda:
            self.logp = self.gen_matrix.cuda()

        out = gumbel_softmax(self.logp, self.temperature, hard)
        if hard:
            hh = torch.zeros((self.del_num * (2 * self.sz - self.del_num - 1) // 2, 2))
            for i in range(out.size()[0]):
                hh[i, out[i]] = 1
            out = hh

        out = out[:, 0]

        if use_cuda:
            out = out.cuda()

        matrix = torch.zeros(self.sz, self.sz).cuda()
        left_mask = torch.ones(self.sz, self.sz)
        left_mask[:-self.del_num, :-self.del_num] = 0
        left_mask = left_mask - torch.diag(torch.diag(left_mask))
        un_index = torch.triu(left_mask).nonzero()

        matrix[(un_index[:, 0], un_index[:, 1])] = out
        out_matrix = matrix + matrix.T
        # out_matrix = out[:, 0].view(self.gen_matrix.size()[0], self.gen_matrix.size()[0])
        return out_matrix

    def init(self, mean, var):
        init.normal_(self.gen_matrix, mean=mean, std=var)


'''Generate unknown part of continuous node state'''
class Generator_states(nn.Module):
    def __init__(self,dat_num,del_num):
        super(Generator_states, self).__init__()
        self.embeddings = nn.Embedding(dat_num, del_num)
    def forward(self, idx):
        pos_probs = torch.sigmoid(self.embeddings(idx)).unsqueeze(2)
        return pos_probs

'''Generate unknown part of discrete node state'''
class Generator_states_discrete(nn.Module):
    def __init__(self,dat_num,del_num):
        super(Generator_states_discrete, self).__init__()
        self.embeddings = nn.Embedding(dat_num, del_num)
    def forward(self, idx):
        pos_probs = torch.sigmoid(self.embeddings(idx)).unsqueeze(2)
        probs = torch.cat([pos_probs, 1 - pos_probs], 2)
        return probs
    
#############
# Functions #
#############
def gumbel_sample(shape, eps=1e-20): # Gumbel Noise
    u = torch.rand(shape)
    gumbel = - np.log(- np.log(u + eps) + eps)
    if use_cuda:
        gumbel = gumbel.to(device)
    return gumbel


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + gumbel_sample(logits.size())
    return torch.nn.functional.softmax(y / temperature, dim=1)



def gumbel_softmax(logits, temperature, hard=False): # sample a row for example
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)  #（N，2）
    if hard:
        #k = logits.size()[-1]   
        y_hard = torch.max(y.data, 1)[1]  #（N，）
        y = y_hard  
    return y




#Tensorboard
#writer = SummaryWriter('../runs/ATGCN'+str(args.delta_t)+'min_new_loss')


############### Eval ######################
# if use cuda
use_cuda = torch.cuda.is_available()

def calc_tptnfpfn(out,adj):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(out.shape[0]):
        for j in range(out.shape[0]):
            if adj[i][j] == 1:
                # positive
                if out[i][j] == 1:
                    # true positive
                    tp += 1
                else:
                    # false positive
                    fn += 1
            else:
                # negative
                if out[i][j] == 1:
                    # true negative
                    fp += 1
                else:
                    # false neg
                    tn += 1
    return tp,tn,fp,fn

def tpr_fpr(out,adj):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(out.shape[0]):
        for j in range(out.shape[0]):
            if adj[i][j] == 1:
                # positive
                if out[i][j] == 1:
                    # true positive
                    tp += 1
                else:
                    # false positive
                    fn += 1
            else:
                # negative
                if out[i][j] == 1:
                    # true negative
                    fp += 1
                else:
                    # false negative
                    tn += 1
    # tpr = tp /  (tp + fp)
    # return tp,tn,fp,fn
    try:
        tpr = float(tp) / (tp + fn)
    except ZeroDivisionError:
        tpr=0
    try:
        fpr = float(fp) / (fp + tn)
    except ZeroDivisionError:
        fpr = 0
    return tpr,fpr

def calc_tpr_fpr(matrix, matrix_pred):
    matrix = matrix.to('cpu').data.numpy()
    matrix_pred = matrix_pred.to('cpu').data.numpy()


    tpr,fpr = tpr_fpr(matrix_pred,matrix)

    return tpr, fpr

def evaluation_indicator(tp,tn,fp,fn):
    try:
        tpr = float(tp) / (tp + fn)
    except ZeroDivisionError:
        tpr=0
    try:
        fpr = float(fp) / (fp + tn)
    except ZeroDivisionError:
        fpr = 0
    try:
        tnr = float(tn) / (tn + fp)
    except ZeroDivisionError:
        tnr = 0
    try:
        fnr = float(fn) / (tp + fn)
    except ZeroDivisionError:
        fnr = 0
    try:
        p = tp / (tp + fp)
    except ZeroDivisionError:
        p = 0
    try:
        r = tp / (tp + fn)
    except ZeroDivisionError:
        r = 0
    try:
        f1_score = 2 * p * r / (p + r)
    except ZeroDivisionError:
        f1_score = 0
    return tpr, fpr, tnr, fnr, f1_score


# eval on dyn
def accuracy(pred, y):
    """
    :param pred: predictions
    :param y: ground truth
    :return: accuracy, defined as 1 - (norm(y - pred) / norm(y))
    """
    return 1 - torch.linalg.norm(y - pred, "fro") / torch.linalg.norm(y, "fro") # use Frobenius norm


def r2(pred, y):
    """
    :param y: ground truth
    :param pred: predictions
    :return: R square (coefficient of determination)
    """

    return 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(y)) ** 2)


def explained_variance(pred, y):
    return 1 - torch.var(y - pred) / torch.var(y)


def cal_dyn_metrics(predictions,y):
   rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
   mae = torchmetrics.functional.mean_absolute_error(predictions, y)
   acc =accuracy(predictions, y)
   r_2 =r2(predictions, y)
   explainedvariance = explained_variance(predictions, y)
   return rmse, mae, acc, r_2, explainedvariance

def dyn_evaluator(predictions,y,batch_size):
    '''
    evaluate for a batch, y.shape=(batchsize,nodesize,dim)
    '''
    predictions=torch.squeeze(predictions)
    y=torch.squeeze(y)

    
    #print(predictions.shape,y.shape)
    rmse, mae, accuracy, r2, explained_variance=cal_dyn_metrics(predictions,y)
    return rmse.item(), mae.item(), accuracy.item(), r2.item(), explained_variance.item()



##########  Read File ################
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
    

def load_TS_parking_ATGCN(batch_size=128,node_num=148,network='ER',delta_t=10,seq_len=3,pre_len=1): 
    # Specially designed for ATGCN
    # Normalized=True

    raw_data=pd.read_csv("/home/sunpeiyan/ParkingTSPrediction/data/SH_Park_"+str(delta_t)+".csv").values #(time_slots,node_num)
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
    train_x=train_x.type(torch.float32) #(,seq_len,node_num)
    train_y=train_y.type(torch.float32)#(,pre_len,node_num)
    test_x=test_x.type(torch.float32)
    test_y=test_y.type(torch.float32)


    train_set=MyDataset(train_x,train_y)
    test_set=MyDataset(test_x,test_y)  # torch.utils.data.random_split
    

    train_loader=DataLoader(train_set,batch_size=batch_size, shuffle=True)
    test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=True)


    edges=pd.read_csv("/home/sunpeiyan/ParkingTSPrediction/data/connect_SH_adj.csv",header=None).values
    return train_loader,test_loader,torch.from_numpy(edges),min_val,max_val




############ Training ########################

def train_dyn_gen(generator, op_net, dyn_isom, op_dyn, config, train_loader, min_val,max_val, loss_fn):

    loss_batch = []
    mse_batch = []
    # Metrics to be Reported
    RMSE=[]
    MAE=[]
    Acc=[]
    R2=[]
    Var=[]
    #print('current temp:', generator.temperature)
    for idx, data in enumerate(train_loader):
        #print('batch idx:', idx)

        # data
        inp,labels=data

        #print(inp.shape)  # (batchsize, seq_len, node_num)
        #print(labels.shape) # (batchsize, pre_len, node_num)

        inp=inp.to(device)
        labels=labels.to(device)

        # drop temperature
        generator.drop_temp()
        outputs = torch.zeros(labels.size(0), labels.size(1), labels.size(2))  # (batch_size,pre_len,node_num)

        # Forward
        op_net.zero_grad()
        op_dyn.zero_grad()

        adj_mat=generator.sample_all(hard=HYP['hard_sample'])
        adj_mat.to(device)
        y_hat = dyn_isom(inp, adj_mat) #(batchsize,pre_len,node_num)


        #loss = torch.mean(torch.abs(y_hat - labels))
        loss=loss_fn(y_hat,labels)

        # Backward
        loss.backward()
        # cut gradient in case nan shows up
        U.clip_grad_norm_(generator.gen_matrix, 0.000075)
        #step
        op_net.step()
        op_dyn.step()

        # calculate metric 
        outputs=y_hat

        outputs= outputs*(max_val-min_val)+min_val  # de-norm
        labels= labels*(max_val-min_val)+min_val

        rmse,mae,acc,r2,var=dyn_evaluator(outputs.cpu(),labels.cpu(),labels.size(0))
        RMSE.append(rmse)
        MAE.append(mae)
        Acc.append(acc)
        R2.append(r2)
        Var.append(var)

        #loss_batch.append(torch.mean(torch.abs(outputs - labels)).item())
        loss_batch.append(loss_fn(outputs,labels).item())
        mse_batch.append(F.mse_loss(labels.cpu(), outputs.cpu()).item())

    # lambda * sum(A_ij)  Regularization
    op_net.zero_grad()
    loss = (torch.sum(generator.sample_all())) * config['lr_stru']
    loss.backward()
    op_net.step()


    # print("RMSE:"+str(np.mean(RMSE)))
    # print("MAE: "+str(np.mean(MAE)))
    # print("Acc: "+str(np.mean(Acc)))
    # print("R2: "+str(np.mean(R2)))
    # print("Var: "+str(np.mean(Var)))

    # tensorboard
    # writer.add_scalar("RMSE",np.mean(RMSE),e)
    # writer.add_scalar("MAE",np.mean(MAE),e)
    # writer.add_scalar("Acc",np.mean(Acc),e)
    # writer.add_scalar("R2",np.mean(R2),e)
    # writer.add_scalar("Var",np.mean(Var),e)

    # each item is the mean of all batches, means this indice for one epoch
    return np.mean(loss_batch), np.mean(mse_batch),np.mean(RMSE), np.mean(MAE), np.mean(Acc), np.mean(R2), np.mean(Var)


def mytrain(config):
    
    if args.sys== 'parking':
        train_loader,test_loader, object_matrix, min_val, max_val= load_TS_parking_ATGCN(batch_size=HYP['batch_size'],node_num=args.nodes,
                                                                           network=args.network,delta_t=HYP['delta_t'],seq_len=HYP['seq_len'],pre_len=HYP['pre_len'])

    # Loss function
    loss_fn = nn.MSELoss()

    # generator
    generator = Gumbel_Generator_Old(sz=args.nodes, temp=HYP['temp'], temp_drop_frac=HYP['drop_frac']).to(device)
    generator.init(0, 0.1)
    adj=generator.sample_all(hard=True)


    # generator optimizer
    op_net = optim.Adam(generator.parameters(), lr=config['lr_net'])
    # dyn learner
    dyn_isom =ATGCN(adj,config["hid"],HYP['dim']).to(device)
    # dyn learner  optimizer
    op_dyn = optim.Adam(dyn_isom.parameters(), lr=config['lr_dyn'])


    best_val_mse = 1000000
    best = 0
    best_loss = 10000000

    for e in range(HYP['epoch_num']):
        # train both dyn learner and generator together
        loss, mse, rmse, mae, acc, r2, var = train_dyn_gen(generator,op_net, dyn_isom, op_dyn, config, train_loader,min_val,max_val, loss_fn)
        if loss < best_loss:       # save model
            print('best epoch:', e)
            best_loss = loss
            best = e
        #print('loss:' + str(loss) + ' mse:' + str(mse))
    print("RMSE:"+str(rmse))
    print("MAE: "+str(mae))
    print("Acc: "+str(acc))
    print("R2: "+str(r2))
    print("Var: "+str(var))
    print(best_loss, best)

    #tune.report(mean_loss=best_loss)
    train.report({'mean_loss': best_loss})


if __name__ == "__main__": 
    # load_data
    
    analysis = tune.run(
        mytrain,
        config={
            "lr_net": tune.sample_from(lambda spec: np.random.uniform(1e-4, 1e-2)),
            "lr_dyn": tune.sample_from(lambda spec: np.random.uniform(1e-4, 1e-2)),
            'lr_stru': tune.sample_from(lambda spec: np.random.uniform(1e-4, 1e-2)),
            'hid':tune.sample_from(lambda spec: np.random.randint(32, 512))
        },
        num_samples=100, #总共运行Trails的数目
        resources_per_trial={"cpu": 12, "gpu": 2},
    )
 
print("Best config: ", analysis.get_best_config(
   metric="mean_loss", mode="min"))
 
# Get a dataframe for analyzing trial results.
df = analysis.results_df
df.to_csv('ray_tune.csv')



