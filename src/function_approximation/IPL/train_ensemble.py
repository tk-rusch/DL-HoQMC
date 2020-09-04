from torch_routines import train
from sys import argv
import numpy as np

def job_id_to_parameters(id,Ns):
    widths = [6, 12, 24]
    depths = [4, 8, 16]
    lrs = [0.0001]
    lambdas = [1e-5,1e-6,1e-7]
    num_inits = 2
    params = []

    for lr in lrs:
        for l in lambdas:
            for w in widths:
                for d in depths:
                    for N in Ns:
                        for i in range(num_inits):
                            params.append(np.array([lr,l,w,d,N]))

    return params[id]

if __name__ == '__main__':
    if(len(argv) > 1):
        id = int(argv[1])
    else:
        id = 1
    Ns = 2**np.arange(7,13)
    max_epochs = 20000
    params = job_id_to_parameters(id-1,Ns)
    train(*params,max_epochs,id)
