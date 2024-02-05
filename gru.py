import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as U

from utils.tools import *
from utils.ReadFile import* 
from utils.eval import *
from torch.utils.tensorboard import SummaryWriter   


# hyperparameters
batch_size=64
l_r=0.003 # learning rate
w_d=1.5e-3 # weight decay
epoch=1500

# Model parameters
seq_len=3
pre_len=1
hidden_dim=100
MODE='Test' # train or test
dataset='COVID'          # choose from {'SH_Park','CN_AQI','Metr-LA','PeMS08','COVID'}

#cuda
torch.cuda.set_device('cuda:0')
device=torch.device("cuda:0")
torch.set_default_dtype(torch.float32)

class GRULinear(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(GRULinear, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.weights = nn.Parameter(
            torch.DoubleTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.DoubleTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [inputs, hidden_state] "[x, h]" (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (batch_size * num_nodes, gru_units + 1)
        concatenation = concatenation.reshape((-1, self._num_gru_units + 1))
        # [x, h]W + b (batch_size * num_nodes, output_dim)


        outputs = concatenation @ self.weights + self.biases
        # [x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # [x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class GRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(GRUCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.linear1 = GRULinear(self._hidden_dim, self._hidden_dim * 2, bias=1.0)
        self.linear2 = GRULinear(self._hidden_dim, self._hidden_dim)

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid([x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.linear1(inputs, hidden_state))
        # r (batch_size, num_nodes * num_gru_units)
        # u (batch_size, num_nodes * num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh([x, (r * h)]W + b)
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.linear2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class GRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, **kwargs):
        super(GRU, self).__init__()
        self._input_dim = input_dim  # num_nodes for prediction
        self._hidden_dim = hidden_dim
        self.gru_cell = GRUCell(self._input_dim, self._hidden_dim)
        self.regressor=nn.Linear(hidden_dim,pre_len).double()

    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        outputs = list()
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        for i in range(seq_len):
            output, hidden_state = self.gru_cell(inputs[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
            outputs.append(output)
        # output (seq_len,batchsize,num_nodes,hidden_dim)
        last_output = outputs[-1]   # (batchsize,num_nodes,hidden_dim)

        # combine regressor part
        last_output=last_output.reshape((-1,last_output.size(2))) #(batchsize*num_nodes,hidden_dim)
        predictions=self.regressor(last_output) # (batchsize*num_nodes,pre_len)
        predictions=predictions.reshape((batch_size, num_nodes, -1)) #(batchsize,num_nodes,pre_len)

        return predictions

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


trainloader,testloader, node_num=load_gru_parking(batch_size,seq_len,pre_len,dataset)
print("Finished preparing dataset")


if MODE=='Train':
    print("start training...")
    # Tensorboard
    writer = SummaryWriter('newruns/GRU_'+dataset)

    MyGRU=GRU(input_dim=node_num,hidden_dim=hidden_dim)
    MyGRU=MyGRU.to(device)

    optimizer = optim.Adam(MyGRU.parameters(), lr=l_r)
    Loss=nn.MSELoss()

    best = 0
    best_loss = 10000000

    for e in range(epoch):

        # Metrics to be evaluated
        RMSE=[]
        MAE=[]
        Acc=[]
        R2=[]
        Var=[]

        print("epoch: "+str(e))

        for batch_id,data in enumerate(trainloader):
            optimizer.zero_grad()

            
            inp,labels=data  # inp:(batchsize,seq_len,node_num)
            inp=inp.to(device)
            labels=labels.to(device)

            predictions=MyGRU(inp) # predictions:(batchsize,node_num,pre_len)
            predictions = predictions.transpose(1, 2).reshape((-1, node_num)) #(batchsize*pre_len,node_num)
            labels = labels.reshape((-1, labels.size(2))) #(batchsize*pre_len,node_num)

            loss=Loss(predictions,labels)

            # backward and optimize
            loss.backward()
       
            #step
            optimizer.step()

            rmse,mae,acc,r2,var=dyn_evaluator(predictions.cpu(),labels.cpu(),labels.size(0))
            RMSE.append(rmse)
            MAE.append(mae)
            Acc.append(acc)
            R2.append(r2)
            Var.append(var)

        # tensorboard
        writer.add_scalar("RMSE",np.mean(RMSE),e)
        writer.add_scalar("MAE",np.mean(MAE),e)
        writer.add_scalar("Acc",np.mean(Acc),e)
        writer.add_scalar("R2",np.mean(R2),e)
        writer.add_scalar("Var",np.mean(Var),e)

        if loss < best_loss:       # save model
            best_loss = loss
            best = e
            dyn_path='newsave/GRU_'+dataset+"_Epoch_"+str(epoch)
            torch.save(MyGRU, dyn_path)
            print('best epoch:', best)
        else:
            print('best epoch:', best)
   
if MODE=='Test':

    MyGRU=GRU(input_dim=node_num,hidden_dim=hidden_dim)
    MyGRU=torch.load('newsave/GRU_'+dataset+"_Epoch_"+str(epoch))
    MyGRU=MyGRU.to(device)

    #optimizer = optim.Adam(MyGRU.parameters(), lr=l_r)
    RMSE=[]
    MAE=[]
    Acc=[]
    R2=[]
    Var=[]
    for batch_id,data in enumerate(testloader):

        #data=data.to(device)
        inp,labels=data  # inp:(batchsize,seq_len,node_num)
        inp=inp.to(device)
        labels=labels.to(device)


        predictions=MyGRU(inp) # predictions:(batchsize,node_num,pre_len)
        predictions = predictions.transpose(1, 2).reshape((-1, node_num)) #(batchsize*pre_len,node_num)
        labels = labels.reshape((-1, labels.size(2))) #(batchsize*pre_len,node_num)

        

        rmse,mae,acc,r2,var=dyn_evaluator(predictions.cpu(),labels.cpu(),labels.size(0))
        RMSE.append(rmse)
        MAE.append(mae)
        Acc.append(acc)
        R2.append(r2)
        Var.append(var)

        
    print("RMSE "+str(np.mean(RMSE)))
    print("MAE  "+str(np.mean(MAE)))
    print("Acc "+str(np.mean(Acc)))
    print("R2 "+str(np.mean(R2)))
    print("Var "+str(np.mean(Var)))




