import pandas as pd
import numpy as np


#aidd_matrix->aidd_adj  threshold=0.95

data=pd.read_csv("aidd_matrix.csv",header=None).values
node_num=len(data)

for i in range(node_num):
    for j in range(node_num):
        if data[i,j]>0.95:
            data[i,j]=1
        else:
            data[i,j]=0

df=pd.DataFrame(data)

df.to_csv('aidd_adj.csv',index=None,header=None)