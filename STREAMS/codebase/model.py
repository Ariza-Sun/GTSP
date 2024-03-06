import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf

import networkx as nx
from tqdm import tqdm
from torch_geometric.nn import GCNConv, GATConv
import plotly.graph_objects as go
from scipy.sparse import coo_matrix
from geopy.distance import distance

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

from utils import *


def init_hidden(x: torch.Tensor, hidden_size: int, num_dir: int = 1, xavier: bool = True):
    """
    Initialize hidden.

    Args:
        x: (torch.Tensor): input tensor
        hidden_size: (int):
        num_dir: (int): number of directions in LSTM
        xavier: (bool): wether or not use xavier initialization
    """
    if xavier:
        return nn.init.xavier_normal_(torch.zeros(num_dir, x.size(0), hidden_size)).to(device)
    return Variable(torch.zeros(num_dir, x.size(0), hidden_size)).to(device)


###########################################################################
################################ ENCODERS #################################
###########################################################################
class AttnEncoder(nn.Module):
    def __init__(self, hidden_size_encoder, seq_len, denoising, directions, input_size, edge_index, spatial,
                 use_spatial=False):
        """
        Initialize the network.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(AttnEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size_encoder
        self.seq_len = seq_len
        self.add_noise = denoising
        self.gcn_e = GCNConv(in_channels=-1, out_channels=self.hidden_size)
        self.directions = directions
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1
        )
        self.attn = nn.Linear(
            in_features=2 * self.hidden_size + self.seq_len,
            out_features=1
        )
        self.softmax = nn.Softmax(dim=1)
        self.edge_index = edge_index
        self.use_spatial = use_spatial
        self.spatial_mat = spatial

    def forward(self, input_data: torch.Tensor, sample: torch.Tensor):
        """
        Forward computation.

        Args:
            input_data: (torch.Tensor): tensor of input data
        """
        spatial = self.spatial_mat[sample]
        spatial = spatial.to(device)
        h_t, c_t = (init_hidden(input_data, self.hidden_size, num_dir=self.directions),
                    init_hidden(input_data, self.hidden_size, num_dir=self.directions))

        # apply the mapping function to the sampled tensor
        a = self.edge_index[0][sample]
        b = self.edge_index[1][sample]
        sample_edge_index = torch.stack([a, b], dim=0)
        mapped_arr = map_values(sample_edge_index, (sample_edge_index.min(), sample_edge_index.max()))

        attentions, input_encoded = (Variable(torch.zeros(input_data.size(0), self.seq_len, self.input_size)),
                                     Variable(torch.zeros(input_data.size(0), self.seq_len, self.hidden_size)))

        if self.add_noise and self.training:
            input_data += self._get_noise(input_data).to(device)

        for t in range(self.seq_len):
            x = torch.cat((h_t.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           c_t.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1).to(device)), dim=2).to(
                device)  # bs * input_size * (2 * hidden_dim + seq_len)

            e_t = self.attn(x.view(-1, self.hidden_size * 2 + self.seq_len))  # (bs * input_size) * 1
            a_t = self.softmax(e_t.view(-1, self.input_size)).to(device)  # (bs, input_size)
            weighted_input = torch.mul(a_t, input_data[:, t, :].to(device))  # (bs * input_size)
            self.lstm.flatten_parameters()
            _, (h_t, c_t) = self.lstm(weighted_input.unsqueeze(0), (h_t, c_t))

            if (self.use_spatial):
                h_t = self.gcn_e(h_t, mapped_arr.to(device))
            input_encoded[:, t, :] = h_t
            attentions[:, t, :] = a_t

        return attentions, input_encoded


###########################################################################
################################ DECODERS #################################
###########################################################################

class AttnDecoder(nn.Module):
    def __init__(self, seq_len, hidden_size_encoder, hidden_size_decoder, output_size, edge_index):
        """
        Initialize the network.

        Args:
            config:
        """
        super(AttnDecoder, self).__init__()
        self.seq_len = seq_len
        self.encoder_hidden_size = hidden_size_encoder
        self.decoder_hidden_size = hidden_size_decoder
        self.out_feats = output_size

        self.gcn_d = GCNConv(in_channels=-1, out_channels=self.decoder_hidden_size)

        self.attn = nn.Sequential(
            nn.Linear(2 * self.decoder_hidden_size + self.encoder_hidden_size, self.encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(self.encoder_hidden_size, 1)
        )
        self.lstm = nn.LSTM(input_size=self.out_feats, hidden_size=self.decoder_hidden_size)
        self.fc = nn.Linear(self.encoder_hidden_size + self.out_feats, self.out_feats)
        self.fc_out = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.out_feats)
        self.fc.weight.data.normal_()
        self.edge_index = edge_index

    def forward(self, input_encoded: torch.Tensor, y_history: torch.Tensor, sample: torch.Tensor):
        """
        Perform forward computation.

        Args:
            input_encoded: (torch.Tensor): tensor of encoded input
            y_history: (torch.Tensor): shifted target
        """
        h_t, c_t = (
        init_hidden(input_encoded, self.decoder_hidden_size), init_hidden(input_encoded, self.decoder_hidden_size))
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        a = self.edge_index[0][sample]
        b = self.edge_index[1][sample]
        sample_edge_index = torch.stack([a, b], dim=0)
        mapped_arr = map_values(sample_edge_index, (sample_edge_index.min(), sample_edge_index.max()))

        for t in range(self.seq_len):
            x = torch.cat((h_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                           c_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                           input_encoded.to(device)), dim=2)
            x = tf.softmax(
                self.attn(
                    x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                ).view(-1, self.seq_len),
                dim=1)
            context = torch.bmm(x.unsqueeze(1), input_encoded.to(device))[:, 0, :]  # (batch_size, encoder_hidden_size)
            y_tilde = self.fc(torch.cat((context.to(device), y_history[:, t].to(device)),
                                        dim=1))  # (batch_size, out_size)
            self.lstm.flatten_parameters()
            _, (h_t, c_t) = self.lstm(y_tilde.unsqueeze(0), (h_t, c_t))
            h_t = self.gcn_d(h_t, mapped_arr.to(device))

        return self.fc_out(torch.cat((h_t[0], context.to(device)), dim=1))  # predicting value at t=self.seq_length+1


class AutoEncForecast(nn.Module):
    def __init__(self, input_att, temporal_att, hidden_size_encoder, seq_len, denoising, directions,
                 hidden_size_decoder, input_size, output_size, sample_size, edge_index, spatial_matrix,
                 use_spatial=False):
        """
        Initialize the network.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(AutoEncForecast, self).__init__()
        self.encoder = AttnEncoder(hidden_size_encoder, seq_len, denoising, directions, input_size, edge_index,
                                   spatial_matrix, use_spatial).to(device) if input_att else \
            Encoder(hidden_size_encoder, seq_len, input_size, edge_index).to(device)
        self.decoder = AttnDecoder(seq_len, hidden_size_encoder, hidden_size_decoder, output_size, edge_index).to(
            device) if temporal_att else Decoder(seq_len, hidden_size_decoder, output_size, edge_index).to(device)

        self.fc1 = nn.Linear(512, 1)  # .cuda()
        self.tanh = nn.Tanh()  # .cuda()
        self.sigmoid = nn.Sigmoid()
        self.sample_size = sample_size

    def calcPerm(self, l: list, m: int):
        for i in permutations(l, m):
            yield list((i))

    def forward(self, encoder_input: torch.Tensor, y_hist: torch.Tensor, sample: torch.Tensor,
                return_attention: bool = False):
        """
        Forward computation. encoder_input_inputs.

        Args:
            encoder_input: (torch.Tensor): tensor of input data
            y_hist: (torch.Tensor): shifted target
            return_attention: (bool): whether or not to return the attention
        """
        attentions, encoder_output = self.encoder(encoder_input, sample)

        outputs = self.decoder(encoder_output, y_hist.float(), sample)
        g = torch.zeros(outputs.shape[1], outputs.shape[1])  # .cuda()
        outputs = outputs.transpose(1, 0)
        idx1 = 0
        idx2 = 0
        for i, j in self.calcPerm(outputs, 2):
            a = self.tanh(self.fc1(torch.cat((i, j))))
            g[idx1, idx2] = a
            if ((idx2 + 1) % self.sample_size):
                idx2 += 1
            else:
                idx2 = 0
                idx1 += 1
        if return_attention:
            return outputs, attentions, g
        return outputs, g
