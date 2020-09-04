import numpy as np

def weighted_function(x,lambda_=0.5):
    dim = x.shape[1]
    j = (np.arange(1,dim+1)**(-2.5)).reshape(dim,1)
    x = np.dot(x,j)
    out = 1./(1.+lambda_*x)
    return out