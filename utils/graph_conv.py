import numpy as np
import scipy.sparse as sp
import torch
from torch.nn.functional import normalize

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def calculate_laplacian_with_self_loop(matrix):
    eye=torch.eye(matrix.size(0)).to(device)
    matrix=matrix.to(device)
    matrix = matrix + eye
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )


    return normalized_laplacian
