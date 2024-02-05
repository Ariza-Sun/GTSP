import torch
import time
import torch.nn.utils as U
import torch.optim as optim

import sys 
sys.path.append("..")     #import from  ../

from model.ATGCN import *
#from utils.tools import *
from utils.ReadFile import* 
from utils.eval import *
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter 
import matplotlib.pyplot as plt  

# choose to use what dataset
dataset='COVID'          # choose from {'SH_Park','CN_AQI','Metr-LA','PeMS08','COVID'}


# config designed for SH_Park.csv
HYP = {
    'hid': 128,  # hidden unit size  #128
    'dim':1,
    'epoch_num': 2000,  # epoch
    'batch_size': 128,  # batch size
    'lr_net': 0.004,  # lr for net generator 0.004   # network structure inference
    'lr_dyn': 0.001,  # lr for dyn learner   0.001      # dynamic reconstruction
    'lr_stru': 0.0001,  # lr for structural loss 0.0001 2000 0.01  0.00001
    'hard_sample': False,  # weather to use hard mode in gumbel
    'sample_time': 1,  # sample time while training
    'temp': 1,  # temperature
    'drop_frac': 1,  # temperature drop frac
    'seq_len':3,            
    'pre_len':1,
    'delta_t':10,  # Time Slice
    'simple': False  # if hard_sample=false, whether to direct sample or gumble soft sample
}

parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=148, help='Number of nodes, default=148')
parser.add_argument('--network', type=str, default='ER', help='type of network')
parser.add_argument('--sys', type=str, default='parking', help='simulated system to model,spring or cmn')
parser.add_argument('--dim', type=int, default=1, help='# information dimension of each node parking:1 ')
parser.add_argument('--delta_t', type=int, default=10, help='time_interval, default=10min')
parser.add_argument('--device_id', type=int, default=1, help='Gpu_id, default=1')
args = parser.parse_args()


start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#load_data
train_loader,test_loader, object_matrix, node_num= load_TS_parking_ATGCN(batch_size=HYP['batch_size'], network=args.network,seq_len=HYP['seq_len'],pre_len=HYP['pre_len'],dataset=dataset)
#object_matrix = object_matrix.cpu().numpy()

# model load path
# if HYP['hard_sample']==True:
#     dyn_path = '../save/ATGCN/dyn_parking' + args.network + '_' + str(args.nodes) + '_dt' + str(HYP['delta_t']) + '.pkl'
#     gen_path = '../save/ATGCN/gen_parking' + args.network + '_' + str(args.nodes) + '_dt' + str(HYP['delta_t']) + '.pkl'
# else:
#     dyn_path = '../save/ATGCN_soft/dyn_parking' + args.network + '_' + str(args.nodes) + '_dt' + str(HYP['delta_t']) + '.pkl'
#     gen_path = '../save/ATGCN_soft/gen_parking' + args.network + '_' + str(args.nodes) + '_dt' + str(HYP['delta_t']) + '.pkl'
dyn_path = '../newsave/ATGCN_'+dataset+'/dyn.pkl'
gen_path = '../newsave/ATGCN_'+dataset+'/gen.pkl'
adj_path = '../newsave/ATGCN_'+dataset+'/adj.pkl'


generator = torch.load(gen_path).to(device)
dyn_isom = torch.load(dyn_path).to(device)

 
pred=[]
label=[]

# Loss function
loss_fn = nn.MSELoss()


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

        # Forward
        adj_mat=generator.sample_all(hard=HYP['hard_sample'],simple=HYP['simple'])
        adj_mat.to(device)
        y_hat = dyn_isom(inp, adj_mat) #(batchsize,pre_len,node_num)

        #loss = torch.mean(torch.abs(y_hat - labels))
        loss=loss_fn(y_hat,labels)

        # calculate metric 
        outputs=y_hat

        # outputs= outputs*(max_val-min_val)+min_val
        # labels= labels*(max_val-min_val)+min_val
        #print(outputs,labels)

        rmse,mae,acc,r2,var=dyn_evaluator(outputs.cpu(),labels.cpu(),labels.size(0))
        RMSE.append(rmse)
        MAE.append(mae)
        Acc.append(acc)
        R2.append(r2)
        Var.append(var)

        #loss_batch.append(torch.mean(torch.abs(outputs - labels)).item())
        loss_batch.append(loss_fn(outputs,labels).item())
        mse_batch.append(F.mse_loss(labels.cpu(), outputs.cpu()).item())

        pred.extend(outputs.cpu().numpy()[:,0,:])
        label.extend(labels.cpu().numpy()[:,0,:])


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
    

# Draw Pictures
pred=np.array(pred)
label=np.array(label)

pred=pred.reshape((-1,1))
label=label.reshape((-1,1))

x=np.arange(0,1)
plt.plot(x,x)

plt.scatter(pred,label,s=0.01)
plt.savefig(dataset+'.png')