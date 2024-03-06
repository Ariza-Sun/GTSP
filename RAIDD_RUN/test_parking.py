import torch
import time
import torch.nn.utils as U
import torch.optim as optim

import sys 
sys.path.append("..")     #import from  ../

from model.rAIDD import *
#from utils.tools import *
from utils.ReadFile import* 
from utils.eval import *
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter   

# config designed for SH_Park.csv
HYP = {
    'node_size': 148,
    'hid': 128,  # hidden unit size
    'dim':1,
    'epoch_num': 200,  # epoch
    'batch_size': 128,  # batch size
    'lr_net': 0.004,  # lr for net generator 0.004   # network structure inference
    'lr_dyn': 0.001,  # lr for dyn learner           # dynamic reconstruction
    'lr_stru': 0.0001,  # lr for structural loss 0.0001 2000 0.01  0.00001
    'hard_sample': False,  # weather to use hard mode in gumbel
    'sample_time': 1,  # sample time while training
    'temp': 1,  # temperature
    'drop_frac': 1,  # temperature drop frac
    'seq_len':12,            
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


start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model load path
dyn_path = '../save/dyn_parking_rnn_Norm_' + args.network + '_' + str(args.nodes) + '_dt' + str(args.delta_t) + '.pkl'
gen_path = '../save/gen_parking_rnn_Norm_' + args.network + '_' + str(args.nodes) + '_dt' + str(args.delta_t) + '.pkl'

generator = torch.load(gen_path).to(device)
dyn_isom = torch.load(dyn_path).to(device)

#load_data
if args.sys== 'parking':
    train_loader,test_loader, object_matrix, min_val, max_val= load_TS_parking(batch_size=HYP['batch_size'],node_num=args.nodes,
                                                                           network=args.network,delta_t=HYP['delta_t'],seq_len=HYP['seq_len'],pre_len=HYP['pre_len'])
object_matrix = object_matrix.cpu().numpy()


def test_dyn_gen():
    loss_batch = []
    mse_batch = []
    RMSE=[]
    MAE=[]
    Acc=[]
    R2=[]
    Var=[]
    print('current temp:', generator.temperature)
    for idx, data in enumerate(test_loader):
        print('batch idx:', idx)
        # data
        inp,labels=data
        inp=inp.to(device)
        labels=labels.to(device)

        # drop temperature
        generator.drop_temp()
        outputs = torch.zeros(labels.size(0), labels.size(1), labels.size(2))  # (batch_size,pre_len,node_num)
        loss_node = []
        for j in range(args.nodes):
            
            # predict and caculate the loss
            adj_col = generator.sample_adj_i(j, hard=HYP['hard_sample'], sample_time=HYP['sample_time']).to(device)

            num = int(args.nodes / HYP['node_size'])
            remainder = int(args.nodes % HYP['node_size'])
            if remainder == 0:
                num = num - 1
            y_hat = dyn_isom(inp, adj_col, j, num, HYP['node_size'],HYP['seq_len'],HYP['pre_len'],HYP['hid']) 
            loss = torch.mean(torch.abs(y_hat - labels[:, :, j]))   

            # use outputs to caculate mse
            outputs[:, :, j] = y_hat

            # record
            loss_node.append(loss.item())

        loss_batch.append(np.mean(loss_node))
        mse_batch.append(F.mse_loss(labels.cpu(), outputs).item())

        #cal metric for a batch 

        outputs= outputs*(max_val-min_val)+min_val
        labels= labels*(max_val-min_val)+min_val
        rmse,mae,acc,r2,var=dyn_evaluator(outputs,labels.cpu(),labels.size(0))
        RMSE.append(rmse)
        MAE.append(mae)
        Acc.append(acc)
        R2.append(r2)
        Var.append(var)

    # each item is the mean of all batches, means this indice for one epoch
    return np.mean(loss_batch), np.mean(mse_batch),np.mean(RMSE),np.mean(MAE),np.mean(Acc),np.mean(R2),np.mean(Var)



with torch.no_grad():
    loss, mse, rmse,mae,acc,r2,var = test_dyn_gen()
    print('loss:' + str(loss) + ' mse:' + str(mse))
    print("RMSE:"+str(rmse))
    print("MAE: "+str(mae))
    print("Acc: "+str(acc))
    print("R2: "+str(r2))
    print("Var: "+str(var))
    


