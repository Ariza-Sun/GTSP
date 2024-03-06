import pandas as pd
import glob
import os
import numpy as np
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from itertools import combinations, permutations

torch.manual_seed(99)
np.random.seed(99)
torch.cuda.empty_cache()
import torch.nn.functional as F


import networkx as nx
from tqdm import tqdm
from torch_geometric.nn import GCNConv, GATConv
import plotly.graph_objects as go
from scipy.sparse import coo_matrix
from geopy.distance import distance

from model import *
from objects import *
from utils import *


# frame = pd.read_csv("C:/Users/psheth5/STCD-RL/data/data/watershed_avg.csv")
frame = pd.read_csv("C:/Users/psheth5/STCD-RL/data/data/StandardizedData.csv")
n_features = frame.shape[1]  
print(frame)

# import numpy as np
# adj_mat = np.load('C:/Users/psheth5/STCD-RL/data/data/watershed.npy')
adj_mat = np.load('ElevationAdjacency.npy')

# # distances = np.load('C:/Users/psheth5/STCD-RL/distance.npy')
# # np.fill_diagonal(distance, np.inf)
# # distances[np.isfinite(distance)] = 1   # replace finite values with 1
# # distances[~np.isfinite(distance)] = 0   # replace infinite values with 0
# # distances


# list_of_coords_str = frame.columns # your list of 3129 latitude longitude values
# list_of_coords = [tuple(map(float, c.split()[::-1])) for c in list_of_coords_str]
# adj_matrix_list = np.zeros((len(list_of_coords), len(frame.columns)))
# for i, c1 in enumerate(list_of_coords):
#     for j, c2 in enumerate(list_of_coords):
#         if c2[0] > c1[0] and distance(c1, c2).km > 0:
#             adj_matrix_list[i][j] = 1


# # adj_mat = np.load('C:/Users/psheth5/STCD-RL/data/data/watershed.npy')
# coo = coo_matrix(adj_matrix_list, dtype = "int8")
coo = coo_matrix(adj_mat, dtype="int8") #转化为稀疏矩阵
row = torch.from_numpy(coo.row.astype(np.int64)).to(torch.long)
col = torch.from_numpy(coo.col.astype(np.int64)).to(torch.long)
edge_index = torch.stack([row, col], dim=0) #（2，num of edges）

# np.save("data\\data\\adj_mat_new.npy", adj_matrix_list)


# edge_index = filter_edge_index(edge_index, list_of_coords)
# edge_index = torch.from_numpy(edge_index.astype(np.int64)).to(torch.long)

task = Tasks('reconstruction')
data1 = TimeSeriesDataset(task=task, data_path="C:/Users/psheth5/STCD-RL/data/data/StandardizedData.csv",
                          categorical_cols=[], index_col=None, target_col=frame.columns[0], seq_length=30,
                          batch_size=256, prediction_window=1)
train_iter, test_iter, nb_features = data1.get_loaders()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

# target_index = 252
target_index = 0

# distance = np.load('C:/Users/psheth5/STCD-RL/data/data/watershed.npy')
# distance = distance[:, target_index]

# spatial_matrix = []

# for d in distance:
#     if np.isfinite(d):
#         spatial_matrix.append(1 / (d + 1))
#     else:
#         spatial_matrix.append(0)

# spatial_matrix1 = torch.FloatTensor(spatial_matrix).cuda()
# torch.max(spatial_matrix1)

spatial_matrix1 = torch.FloatTensor(adj_mat).cuda()
torch.max(spatial_matrix1)

sampling_size = 100
model = AutoEncForecast(input_att=True, temporal_att=True, hidden_size_encoder=64, seq_len=30, denoising=False,
                        directions=1, hidden_size_decoder=64, input_size=sampling_size, output_size=sampling_size,
                        sample_size=sampling_size, edge_index=edge_index, spatial_matrix=spatial_matrix1,
                        use_spatial=True).to(device)

############################## Training ######################################
random.seed(26)
np.random.seed(26)
torch.manual_seed(26)
torch.cuda.manual_seed(26)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
global_step, logging_loss = 0, 0.0
train_loss = 0.0
reg1 = True
reg2 = True
reg_factor1 = 1e-8
reg_factor2 = 1e-8
gradient_accumulation_steps = 1
max_grad_norm = 0.1
logging_steps = 100
criterion = nn.BCELoss()
crit_mse = nn.MSELoss()
crit_rss = nn.MSELoss(reduction='none')
crit_nll = nn.NLLLoss()
save_steps = 5000
eval_during_training = False
output_dir = "Models"
lrs_step_size = 5000
do_eval = True
reg_factor = 1
BASELINE_REWARD = 20
breakout = 0
inferred_cg = np.zeros((n_features, n_features))
attention = np.zeros((n_features, n_features))
min_loss = 1e+5
for epoch in tqdm(range(5), unit="epoch"):
    attentions = []
    outputs = []
    total_loss = 0
    loss_1 = []
    loss_2 = []
    for i, batch in tqdm(enumerate(train_iter), total=len(train_iter), unit="batch"):
        original_list = list(range(3129))
        while (len(original_list) != 0):
            sampled_list = []
            optimizer.zero_grad()
            model.train()
            feature, y_hist, target = batch
            try:
                sample = random.sample(original_list, sampling_size)
            except:
                sample = random.sample(original_list, len(original_list))
                sample.extend(random.sample(range(feature.shape[2]), sampling_size - len(original_list) - 1))
                sample.append(0)
            # add the sampled elements to the sampled list
            sampled_list.extend(sample)
            # remove the sampled elements from the original list
            original_list = [element for element in original_list if element not in sample]
            feature = feature[:, :, sample]
            y_hist = y_hist[:, :, sample]
            target = target[:, sample]

            output, att, g = model(feature.to(device), y_hist.to(device), sample, return_attention=True)
            temp = torch.mean(att, axis=1)
            output = output.reshape(output.shape[1], output.shape[0])
            inferred_cg = update_graph(sample, inferred_cg, g)
            inf_cg = torch.FloatTensor(inferred_cg)
            attention = update_attention(sample, att, attention)
            num_params = count_parameters(model)
            # target = target.cuda()
            loss2 = crit_mse(output, target)

            n = torch.FloatTensor([feature.shape[0]])  # .cuda()
            cyc = torch.trace(torch.matrix_exp(inf_cg * inf_cg)) - 3129
            bic = n * 3129 * (torch.log(torch.sum(crit_rss(output, target)) / (3129 * n))) + torch.sum(
                inf_cg) * torch.log(n)
            loss = -1 * (loss2 + reg_factor1 * cyc + reg_factor2 * bic)
            loss_2.append(loss2)
            loss_1.append(loss.item())
            if (len(original_list) % 1000 == 0):
                print("The number of nodes sampled are ", 3129 - len(original_list))
    loss_tot = 0
    loss_21 = calc_discounted_rewards(loss_2, 0.98)
    for indexe in range(len(loss_21)):
        loss_tot += loss_1[indexe] * torch.log(loss_21[indexe])
    loss_tot = (loss_tot - BASELINE_REWARD) / len(loss_21)
    loss_tot.backward()
    optimizer.step()
    total_loss += loss_tot.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if (round(total_loss / (i + 1), 2) <= round(min_loss / (i + 1), 2)):
        min_loss = total_loss
        breakout = 0
        torch.save(model.state_dict(), os.getcwd() + "/Models/MyModel_GCN_FullModel_" + str(epoch))
    else:
        breakout += 1
    print("Loss is ", total_loss, " for epoch ", epoch)
    print(np.where(attention[target_index, :] != 0), len(np.where(attention[target_index, :] != 0)[0]),
          get_num_cycles(inferred_cg), is_dag(inferred_cg))


################################### Evaluation ######################
def plot_graph(a_new, fname):
    latitude = []
    longitude = []
    for j, i in enumerate(a_new):
        latitude.append(float(i.split()[0]))
        longitude.append(float(i.split()[1]))
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lat=longitude,
        lon=latitude,
        marker={'color': 'red',
                'size': 10}
    ))

    fig.update_layout(margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
                      mapbox={
                          'center': {'lon': 139, 'lat': 36.5},
                          'style': "stamen-terrain",
                          'zoom': 4.5},
                      width=1600,
                      height=900, )
    fig.show()
    fig.write_html(fname)


at1 = frame.columns[inferred_cg[target_index, :].argsort()[-100:]]
plot_graph(at1, "mapTestFT1.html")

np.save("attention.npy", attention)
np.save("inferred_cg.npy", inferred_cg)