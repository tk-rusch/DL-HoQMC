import torch
from torch import nn, optim
import numpy as np
from pathlib import Path
import os

def HO_loss(x1, y1, x2, y2):
    loss = nn.MSELoss()
    return 2.*loss(x1, y1) + loss(x2, y2)

def E_T(x1, y1, x2, y2):
    loss = nn.MSELoss()
    return torch.sqrt(torch.abs(2.*loss(x1, y1) - loss(x2, y2)))

def init_weights(m):
    for name, param in m.named_parameters():
        if('weight' in name):
            torch.nn.init.xavier_normal_(param)

def get_neural_network(layer_sizes, activationFunction, Xavier_init=True):
    model = nn.Sequential()
    model.add_module('Transformation_1', nn.Linear(layer_sizes[0], layer_sizes[1]))
    model.add_module('sigma_1', activationFunction())
    for i in range(2, len(layer_sizes) - 1):
        model.add_module('Transformation_'+str(i),nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        model.add_module('sigma_'+str(i),activationFunction())
    model.add_module('Transformation_'+str(len(layer_sizes) -1),nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    if(Xavier_init):
        init_weights(model)
    return model

def train(lr,reg,width,depth,N,max_epochs,id):
    ninp = 16
    id = 1
    id2 = id
    width = int(width)
    depth = int(depth)

    layer_sizes = [ninp]
    for i in range(depth - 1):
        layer_sizes.append((width))
    layer_sizes.append(1)
    model = get_neural_network(layer_sizes, nn.Tanh)

    cwd = os.getcwd()
    data_path = cwd + '/../../../data/elliptic/discretization_test'

    train_x1 = torch.from_numpy(np.loadtxt(data_path+'/h_'+str(id)+'/In_'+str(int(np.log2(N)))+'.txt',delimiter=',').T)
    train_y1 = torch.from_numpy(np.loadtxt(data_path+'/h_'+str(id)+'/Out'+str(int(np.log2(N)))+'.txt',skiprows=1)[:,1]).unsqueeze(-1)

    train_x2 = torch.from_numpy(np.loadtxt(data_path+'/h_'+str(id)+'/In_'+str(int(np.log2(N)-1))+'.txt',delimiter=',').T)
    train_y2 = torch.from_numpy(np.loadtxt(data_path+'/h_'+str(id)+'/Out'+str(int(np.log2(N)-1))+'.txt',skiprows=1)[:,1]).unsqueeze(-1)

    test_x = torch.from_numpy(np.loadtxt(data_path+'/h_'+str(id2)+'/In_12.txt',delimiter=',')[:,1:].T)
    test_y = torch.from_numpy(np.loadtxt(data_path+'/h_'+str(id2)+'/Out12.txt',skiprows=1)[1:,1]).unsqueeze(-1)

    optimizer = optim.Adam(model.parameters(), lr, weight_decay=reg)
    epochs = max_epochs

    def test(x,y):
        model.eval()
        with torch.no_grad():
            test_loss = nn.MSELoss()
            output = model(x.float())
            loss = torch.sqrt(test_loss(output,y.float()))
        return loss.item()

    for e in range(epochs):
        model.train()
        optimizer.zero_grad()
        output1 = model(train_x1.float())
        output2 = model(train_x2.float())
        loss = HO_loss(output1, train_y1.float(), output2, train_y2.float())
        loss.backward()
        optimizer.step()

    output1 = model(train_x1.float())
    output2 = model(train_x2.float())
    train_loss = E_T(output1, train_y1.float(), output2, train_y2.float()).item()
    test_loss = test(test_x, test_y)

    Path('new_results_'+str(id)).mkdir(parents=True, exist_ok=True)
    with open('new_results_'+str(id)+'/N_'+str(int(N))+'.txt','a') as file:
        file.write(str(train_loss)+ ' ' + str(test_loss)+'\n')