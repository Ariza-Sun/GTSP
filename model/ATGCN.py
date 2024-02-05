import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np

import sys
sys.path.append("..")  
from utils.graph_conv import calculate_laplacian_with_self_loop
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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



class TGCNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int): #num_gru_units=hidden_dim
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        #self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )    # r, u 
        self.graph_conv2 = TGCNGraphConvolution(
             self._hidden_dim, self._hidden_dim
        )    # c

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



#####################
# Network Generator #
#####################


class Gumbel_Generator_Old(nn.Module):  
    def __init__(self, sz=10, temp=10, temp_drop_frac=0.9999):
        super(Gumbel_Generator_Old, self).__init__()
        self.sz = sz  # N
        self.gen_matrix = Parameter(torch.rand(sz, sz, 2)) 
        self.temperature = temp
        self.temp_drop_frac = temp_drop_frac
        self.ReLU=nn.ReLU()

    def drop_temp(self):
        # drop temperature
        self.temperature = self.temperature * self.temp_drop_frac

    # output: a matrix
    def sample_all(self, hard=False, simple=False):

        if simple==True and hard==False:
            out_matrix=self.gen_matrix[:,:,0]
            out_matrix=self.ReLU(out_matrix)

        else:    
            self.logp = self.gen_matrix.view(-1, 2)  #(N*N, 2)
            out = gumbel_softmax(self.logp, self.temperature, hard) # different shape for hard seems
            if hard:
                hh = torch.zeros(self.gen_matrix.size()[0] ** 2, 2)
                for i in range(out.size()[0]):  # N*N
                    hh[i, out[i]] = 1
                out = hh  # the larger dim becomes 1
            if use_cuda:
                out = out.cuda()

            # use dim=0
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



def gumbel_softmax(logits, temperature, hard=False): # return: log(\theta)+ gumbel
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
    y = gumbel_softmax_sample(logits, temperature)  #（N*N，2）

    if hard:
        #k = logits.size()[-1]   
        y_hard = torch.max(y.data, 1)[1]  #（N*N, ） #最大位置的索引
        y = y_hard  
    return y




