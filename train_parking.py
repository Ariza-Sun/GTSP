import torch
import time
import torch.nn.utils as U
import torch.optim as optim
from model.AIDD import *
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
    'hid': 128,  # hidden unit size
    'epoch_num': 500,  # epoch
    'batch_size': 128,  # batch size
    'lr_net': 0.004,  # lr for net generator 0.004   # network structure inference
    'lr_dyn': 0.001,  # lr for dyn learner           # dynamic reconstruction
    'lr_stru': 0.0001,  # lr for structural loss 0.0001 2000 0.01  0.00001
    'hard_sample': False,  # weather to use hard mode in gumbel
    'sample_time': 1,  # sample time while training
    'temp': 1,  # temperature
    'drop_frac': 1,  # temperature drop frac
}


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, default='ER', help='type of network')
parser.add_argument('--sys', type=str, default='parking', help='simulated system to model,spring or cmn')
parser.add_argument('--dim', type=int, default=1, help='# information dimension of each node parking:1 ')
parser.add_argument('--device_id', type=int, default=1, help='Gpu_id, default=1')
args = parser.parse_args()

#set gpu id
#print(torch.cuda.is_available())
torch.cuda.set_device(args.device_id)
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load_data
train_loader, val_loader, test_loader, object_matrix, node_num = load_parking(batch_size=HYP['batch_size'],network=args.network,dataset=dataset)
HYP['node_size']=node_num
#object_matrix = object_matrix.cpu().numpy()


#Tensorboard
writer = SummaryWriter('newruns/AIDD_'+dataset)

# generator
generator = Gumbel_Generator_Old(sz=node_num, temp=HYP['temp'], temp_drop_frac=HYP['drop_frac']).to(device)
generator.init(0, 0.1)
# generator optimizer
op_net = optim.Adam(generator.parameters(), lr=HYP['lr_net'])

# dyn learner
dyn_isom = IO_B(args.dim, HYP['hid']).to(device)
# dyn learner  optimizer
op_dyn = optim.Adam(dyn_isom.parameters(), lr=HYP['lr_dyn'])




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
        data = data.to(device)
        x = data[:, :, 0, :] #(batchsize,node,dim)
        y = data[:, :, 1, :]
        # drop temperature
        generator.drop_temp()
        outputs = torch.zeros(y.size(0), y.size(1), y.size(2))  # (batch_size,node_size,dim)
        loss_node = []
        for j in range(node_num):
            # zero grad
            op_net.zero_grad()
            op_dyn.zero_grad()
            # predict and caculate the loss
            adj_col = generator.sample_adj_i(j, hard=HYP['hard_sample'], sample_time=HYP['sample_time']).to(device)

            num = int(node_num/ HYP['node_size'])
            remainder = int(node_num % HYP['node_size'])
            if remainder == 0:
                num = num - 1
            # Forward
            y_hat = dyn_isom(x, adj_col, j, num, HYP['node_size'])  # (batchsize,dim)
            loss = torch.mean(torch.abs(y_hat - y[:, j, :]))
            # backward and optimize
            loss.backward()
            # cut gradient in case nan shows up
            U.clip_grad_norm_(generator.gen_matrix, 0.000075)
            #step
            op_net.step()
            op_dyn.step()

            # use outputs to caculate mse    # only use y_hat of Node j
            outputs[:, j, :] = y_hat

            # record
            loss_node.append(loss.item())

        #cal metric for a batch 
        rmse,mae,acc,r2,var=dyn_evaluator(outputs,y.cpu(),y.size(0))
        RMSE.append(rmse)
        MAE.append(mae)
        Acc.append(acc)
        R2.append(r2)
        Var.append(var)

        loss_batch.append(np.mean(loss_node))
        mse_batch.append(F.mse_loss(y.cpu(), outputs).item())

    # lambda * sum(A_ij)
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
dyn_path = 'newsave/AIDD_'+dataset+'/dyn.pkl'
gen_path = 'newsave/AIDD_'+dataset+'/gen.pkl'
adj_path = 'newsave/AIDD_'+dataset+'/adj.pkl'

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
