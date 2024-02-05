import torch
import torchmetrics
import numpy as np
import pandas as pd
import os
import math
from sklearn.metrics import roc_curve, auc

# if use cuda
use_cuda = torch.cuda.is_available()

def calc_tptnfpfn(out,adj):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(out.shape[0]):
        for j in range(out.shape[0]):
            if adj[i][j] == 1:
                # positive
                if out[i][j] == 1:
                    # true positive
                    tp += 1
                else:
                    # false positive
                    fn += 1
            else:
                # negative
                if out[i][j] == 1:
                    # true negative
                    fp += 1
                else:
                    # false neg
                    tn += 1
    return tp,tn,fp,fn

def tpr_fpr(out,adj):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(out.shape[0]):
        for j in range(out.shape[0]):
            if adj[i][j] == 1:
                # positive
                if out[i][j] == 1:
                    # true positive
                    tp += 1
                else:
                    # false positive
                    fn += 1
            else:
                # negative
                if out[i][j] == 1:
                    # true negative
                    fp += 1
                else:
                    # false negative
                    tn += 1
    # tpr = tp /  (tp + fp)
    # return tp,tn,fp,fn
    try:
        tpr = float(tp) / (tp + fn)
    except ZeroDivisionError:
        tpr=0
    try:
        fpr = float(fp) / (fp + tn)
    except ZeroDivisionError:
        fpr = 0
    return tpr,fpr

def calc_tpr_fpr(matrix, matrix_pred):
    matrix = matrix.to('cpu').data.numpy()
    matrix_pred = matrix_pred.to('cpu').data.numpy()


    tpr,fpr = tpr_fpr(matrix_pred,matrix)

    return tpr, fpr

def evaluation_indicator(tp,tn,fp,fn):
    try:
        tpr = float(tp) / (tp + fn)
    except ZeroDivisionError:
        tpr=0
    try:
        fpr = float(fp) / (fp + tn)
    except ZeroDivisionError:
        fpr = 0
    try:
        tnr = float(tn) / (tn + fp)
    except ZeroDivisionError:
        tnr = 0
    try:
        fnr = float(fn) / (tp + fn)
    except ZeroDivisionError:
        fnr = 0
    try:
        p = tp / (tp + fp)
    except ZeroDivisionError:
        p = 0
    try:
        r = tp / (tp + fn)
    except ZeroDivisionError:
        r = 0
    try:
        f1_score = 2 * p * r / (p + r)
    except ZeroDivisionError:
        f1_score = 0
    return tpr, fpr, tnr, fnr, f1_score


# eval on the generator

def constructor_evaluator(gumbel_generator, tests, obj_matrix,e):

    err_list = []
    tp_list=[]
    tn_list = []
    fp_list = []
    fn_list = []
    tpr_list = []
    fpr_list = []
    tnr_list = []
    fnr_list = []
    f1_list=[]
    auc_list=[]

    obj_matrix = torch.from_numpy(obj_matrix)  # obj_matrix is ground truth of Θ
    for t in range(tests):  # 'tests' is for the iter number of tests

        mat = gumbel_generator.gen_matrix.cpu().view(-1, 2) 
        y_score1 = torch.nn.functional.softmax(mat, dim=1)[:, 0].detach().numpy()
        fpr, tpr, threshold = roc_curve(obj_matrix.numpy().reshape(-1), y_score1) # eval Θ
        soft_auc = auc(fpr, tpr)

        out_matrix = gumbel_generator.sample_all(hard=True, epoch=e) # hard version Θ
        out_matrix = out_matrix.cpu()
        err = torch.sum(torch.abs(out_matrix - obj_matrix))

        err = err.cpu() if use_cuda else err
        # if we got nan in err
        if math.isnan(err):
            print('problem cocured')
            # torch.save(gumbel_generator,'problem_generator_genchange.model')
            d()
            t=t-1
            continue
        err_list.append(err.data.numpy().tolist())
        tp, tn, fp, fn = calc_tptnfpfn(out_matrix,obj_matrix)
        tpr, fpr, tnr, fnr, f1_score = evaluation_indicator(tp,tn,fp,fn)
        tp_list.append(tp)
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        tnr_list.append(tnr)
        fnr_list.append(fnr)
        f1_list.append(f1_score)
        auc_list.append(soft_auc)

    print('err:', np.mean(err_list))
    print('tp:', np.mean(tp_list))
    print('tn:', np.mean(tn_list))
    print('fp:', np.mean(fp_list))
    print('fn:', np.mean(fn_list))
    print('tpr:', np.mean(tpr_list))
    print('fpr:', np.mean(fpr_list))
    print('tnr:', np.mean(tnr_list))
    print('fnr:', np.mean(fnr_list))
    print('f1:', np.mean(f1_list))
    print('auc:', np.mean(auc_list))


# eval on dyn
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
    evaluate for a batch, y.shape=(batchsize,nodesize,dim)
    '''
    predictions=torch.squeeze(predictions)
    y=torch.squeeze(y)

    
    #print(predictions.shape,y.shape)
    rmse, mae, accuracy, r2, explained_variance=cal_dyn_metrics(predictions,y)
    return rmse.item(), mae.item(), accuracy.item(), r2.item(), explained_variance.item()