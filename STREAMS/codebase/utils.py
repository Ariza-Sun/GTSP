import numpy as np
import torch
import networkx as nx
import torchmetrics


def filter_edge_index(edge_index, coords):
    filtered_index = []
    for i in range(edge_index.shape[1]):
        u = edge_index[0][i]
        v = edge_index[1][i]
        # check if u is above v
        if torch.tensor(coords[u][0]) > torch.tensor(coords[v][0]) and u != v:
            filtered_index.append([u, v])
    filtered_index = np.array(filtered_index).T
    return filtered_index


# create a mapping function
def map_values(value, value_range):
    """Maps a value from the input range to the output range."""
    input_min, input_max = value_range
    output_min, output_max = (0, 99)
    return torch.floor(((value - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min).long()


def reverse_map_values(mapped_value, value_range):
    """Maps a value from the output range to the input range."""
    input_min, input_max = value_range
    output_min, output_max = (0, 99)
    return torch.floor(
        ((mapped_value - output_min) / (output_max - output_min)) * (input_max - input_min) + input_min).long()


import random


def update_graph(sample, inferred_cg, g):
    for i in range(len(sample)):
        for j in range(len(sample)):
            inferred_cg[sample[i], sample[j]] += g[i, j].cpu()
    return inferred_cg


def update_attention(sample, att, attention):
    temp = torch.mean(att, axis=1)
    for i in range(len(sample)):
        for j in range(len(sample)):
            attention[sample[i], sample[j]] += temp[i, j].cpu()
    return attention


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_num_cycles(gra):
    # gra = gra.cpu().detach().numpy()
    G = nx.from_numpy_array(gra) # former: matrix
    H = G.to_directed()
    try:
        return torch.FloatTensor([len(list(nx.find_cycle(H, orientation="original")))])  # .cuda()
    except:
        return 0


def is_dag(gr):
    # gr = gr.cpu().detach().numpy()
    G = nx.from_numpy_array(gr)
    return int(nx.is_directed_acyclic_graph(G))


## helper functions
def calc_discounted_rewards(rewards, gamma):
    '''
    Simple implementation for better understanding
    gets rewards of an entire episode and calculates R_t for every t
    '''

    returns = []

    for t in range(len(rewards)):
        ret = 0

        for t_p in range(t, len(rewards)):
            ret += gamma ** (t_p - t) * rewards[t_p]

        returns.insert(0, ret)

    return returns


# My eval on dyn
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
    evaluate for a batch, y.shape=(batchsize,nodesize, dim)
    '''
    predictions=torch.squeeze(predictions)
    y=torch.squeeze(y)

    
    #print(predictions.shape,y.shape)
    rmse, mae, accuracy, r2, explained_variance=cal_dyn_metrics(predictions,y)
    return rmse.item(), mae.item(), accuracy.item(), r2.item(), explained_variance.item()