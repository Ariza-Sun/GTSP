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


# choose to use what dataset
dataset='COVID'          # choose from {'SH_Park','CN_AQI','Metr-LA','PeMS08','COVID'}

# config designed for SH_Park.csv
HYP = {
    'hid': 128,  # hidden unit size #128
    'dim':1,
    'epoch_num': 100,  # epoch
    'batch_size': 128,  # batch size
    'lr_net': 0.004,  # lr for net generator 0.004   # network structure inference  ####revised
    'lr_dyn': 0.001,  # lr for dyn learner           # dynamic reconstruction
    'lr_stru': 0.0001,  # lr for structural loss 0.0001 2000 0.01  0.00001
    'hard_sample': False,  # weather to use hard mode in gumbel
    'sample_time': 1,  # sample time while training
    'temp': 1,  # temperature
    'drop_frac':1,  # temperature drop frac: 1 means no drop
    'seq_len':3,            
    'pre_len':1,
    'delta_t':10,   # Time Slice
    'simple':False # if hard_sample=false, whether to direct sample or gumble sample
}


parser = argparse.ArgumentParser()
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

# Load Data
train_loader,test_loader, object_matrix, node_num= load_TS_parking_ATGCN(batch_size=HYP['batch_size'], network=args.network,seq_len=HYP['seq_len'],pre_len=HYP['pre_len'],dataset=dataset)

#Tensorboard
writer = SummaryWriter('../newruns/ATGCN_'+dataset)

# generator
generator = Gumbel_Generator_Old(sz=node_num, temp=HYP['temp'], temp_drop_frac=HYP['drop_frac']).to(device)
if HYP['simple']==True:
    generator.init(0.5,0.1)
else:
    generator.init(0, 0.1)
adj=generator.sample_all(hard=HYP['hard_sample'],simple=HYP['simple'])


# generator optimizer
op_net = optim.Adam(generator.parameters(), lr=HYP['lr_net'])

# dyn learner
dyn_isom =ATGCN(adj,HYP["hid"],HYP['dim']).to(device)

# dyn learner  optimizer
op_dyn = optim.Adam(dyn_isom.parameters(), lr=HYP['lr_dyn'])

# Loss function
loss_fn = nn.MSELoss()


def train_dyn_gen(e):
    loss_batch = []
    mse_batch = []
    # Metrics to be Reported
    RMSE=[]
    MAE=[]
    Acc=[]
    R2=[]
    Var=[]
    print('current temp:', generator.temperature)
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

        adj_mat=generator.sample_all(hard=HYP['hard_sample'],simple=HYP['simple'])
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

        # outputs= outputs*(max_val-min_val)+min_val  # de-norm
        # labels= labels*(max_val-min_val)+min_val

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
    loss = (torch.sum(generator.sample_all())) * HYP['lr_stru']
    loss.backward()
    op_net.step()


    # tensorboard
    writer.add_scalar("RMSE",np.mean(RMSE),e)
    writer.add_scalar("MAE",np.mean(MAE),e)
    writer.add_scalar("Acc",np.mean(Acc),e)
    writer.add_scalar("R2",np.mean(R2),e)
    writer.add_scalar("Var",np.mean(Var),e)

    # each item is the mean of all batches, means this indice for one epoch
    return np.mean(loss_batch), np.mean(mse_batch)


# start training
best_val_mse = 1000000
best = 0
best_loss = 10000000

# model save path
# if HYP['hard_sample']==True:
#     dyn_path = '../save/ATGCN/dyn_parking' + args.network + '_' + str(args.nodes) + '_dt' + str(HYP['delta_t']) + '.pkl'
#     gen_path = '../save/ATGCN/gen_parking' + args.network + '_' + str(args.nodes) + '_dt' + str(HYP['delta_t']) + '.pkl'
#     adj_path = '../save/ATGCN/adj_parking' + args.network + '_' + str(args.nodes) + '_dt' + str(HYP['delta_t']) + '.pkl'

# else:
#     dyn_path = '../save/ATGCN_soft/dyn_parking' + args.network + '_' + str(args.nodes) + '_dt' + str(HYP['delta_t']) + '.pkl'
#     gen_path = '../save/ATGCN_soft/gen_parking' + args.network + '_' + str(args.nodes) + '_dt' + str(HYP['delta_t']) + '.pkl'
#     adj_path = '../save/ATGCN_soft/adj_parking' + args.network + '_' + str(args.nodes) + '_dt' + str(HYP['delta_t']) + '.pkl'
dyn_path = '../newsave/ATGCN_'+dataset+'/dyn.pkl'
gen_path = '../newsave/ATGCN_'+dataset+'/gen.pkl'
adj_path = '../newsave/ATGCN_'+dataset+'/adj.pkl'



# Main Function
# each training epoch
for e in range(HYP['epoch_num']):
    print('\nepoch', e)
    t_s = time.time()
    t_s1 = time.time()
    try:
        # train both dyn learner and generator together
        loss, mse = train_dyn_gen(e)

    except RuntimeError as sss:
        if 'out of memory' in str(sss):
            print('|WARNING: ran out of memory')
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise sss

    t_e1 = time.time()
    print('loss:' + str(loss) + ' mse:' + str(mse))
    print('time for this dyn_adj epoch:' + str(round(t_e1 - t_s1, 2)))

    if loss < best_loss:       # save model
        print('best epoch:', e)
        best_loss = loss
        best = e
        torch.save(dyn_isom, dyn_path)
        torch.save(generator, gen_path)
        out_matrix = generator.sample_all(hard=HYP['hard_sample'], ).to(device)
        torch.save(out_matrix, adj_path)
    print('best epoch:', best)
    # if e > 1:
    t_s2 = time.time()

    #Evaluate the accuracy of predict adj (couldn't do because don't have ground truth)
    #constructor_evaluator(generator, 1, np.float32(object_matrix), e)

    t_e2 = time.time()
    #print('time for this adj_eva epoch:' + str(round(t_e2 - t_s2, 2)))
    t_e = time.time()
    print('time for this whole epoch:' + str(round(t_e - t_s, 2)))

end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('end_time:', end_time)
writer.close()
