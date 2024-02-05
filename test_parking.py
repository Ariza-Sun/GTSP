import torch
import time
from model.AIDD import *
#from tools import *
from utils.ReadFile import* 
from utils.eval import *
import argparse


# choose to use what dataset
dataset='COVID'          # choose from {'SH_Park','CN_AQI','Metr-LA','PeMS08','COVID'}


# configuration
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
#parser.add_argument('--nodes', type=int, default=148, help='Number of nodes, default=148')
parser.add_argument('--network', type=str, default='ER', help='type of network')
parser.add_argument('--sys', type=str, default='parking', help='simulated system to model,spring or cmn')
parser.add_argument('--dim', type=int, default=1, help='# information dimension of each node parking:1 ')
parser.add_argument('--device_id', type=int, default=1, help='Gpu_id, default=1')
args = parser.parse_args()

start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load_data
train_loader, val_loader, test_loader, object_matrix, node_num = load_parking(batch_size=HYP['batch_size'],network=args.network,dataset=dataset)
HYP['node_size']=node_num
#object_matrix = object_matrix.cpu().numpy()

# model load path
dyn_path = 'newsave/AIDD_'+dataset+'/dyn.pkl'
gen_path = 'newsave/AIDD_'+dataset+'/gen.pkl'

generator = torch.load(gen_path).to(device)
dyn_isom = torch.load(dyn_path).to(device)





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
        data = data.to(device)
        x = data[:, :, 0, :]
        y = data[:, :, 1, :]

        if idx==0:  # vis
            y_all_label=y.cpu()
        else:
            y_all_label=torch.cat((y_all_label,y.cpu()),0)
 
        # drop temperature
        generator.drop_temp()
        outputs = torch.zeros(y.size(0), y.size(1), y.size(2))
        loss_node = []
        for j in range(node_num):
            
            # predict and caculate the loss
            adj_col = generator.sample_adj_i(j, hard=HYP['hard_sample'], sample_time=HYP['sample_time']).to(device)

            num = int(node_num / HYP['node_size'])
            remainder = int(node_num % HYP['node_size'])
            if remainder == 0:
                num = num - 1
            y_hat = dyn_isom(x, adj_col, j, num, HYP['node_size'])
            loss = torch.mean(torch.abs(y_hat - y[:, j, :]))
            # use outputs to caculate mse
            outputs[:, j, :] = y_hat

            # record
            loss_node.append(loss.item())

        if idx==0:  # vis
            y_all_pred=outputs.cpu()
        else:
            y_all_pred=torch.cat((y_all_pred,outputs.cpu()),0)
 

        loss_batch.append(np.mean(loss_node))
        mse_batch.append(F.mse_loss(y.cpu(), outputs).item())

        #cal metric for a batch 
        rmse,mae,acc,r2,var=dyn_evaluator(outputs,y.cpu(),y.size(0))
        RMSE.append(rmse)
        MAE.append(mae)
        Acc.append(acc)
        R2.append(r2)
        Var.append(var)

    #### save prediction values in file
    y_all_label=y_all_label.squeeze()
    y_all_pred=y_all_pred.squeeze()

    # np.savetxt('y_all_pred.csv',y_all_pred.numpy())
    # np.savetxt('y_all_label.csv',y_all_label.numpy())

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
    


