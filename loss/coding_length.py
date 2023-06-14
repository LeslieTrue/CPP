import torch
import torch.nn as nn
import numpy as np
class CodingLength(nn.Module):
    def __init__(self, eps=0.01):
        super(CodingLength, self).__init__()
        self.eps = eps
        print('eps: ', self.eps)
    def forward(self, X):
        #normalize over the dim_heads dimension
        '''
        X with shape (m, n): m samples with n dimensions
        V with shape (n, m): n dimensions with m samples
        '''
        m, n = X.shape
        V = X.T
        scalar = n/(self.eps * m)
        
        product = V.matmul(V.T)
        I = torch.eye(n)
        logdet = torch.logdet(I + scalar * product)
        return (m+n)*logdet/2.
def seg_compute(feature, membership,num_clusters):
    total = 0
    criterion = CodingLength(eps=0.1)

    for i in range(num_clusters):
        idx = (membership == i)
        if idx.sum() == 0:
            continue
        X = feature[idx]
        # print(X.shape[0], feature.shape[0])
        total += criterion(X) + X.shape[0] * (-torch.log(torch.tensor(X.shape[0]/feature.shape[0])))
    return total