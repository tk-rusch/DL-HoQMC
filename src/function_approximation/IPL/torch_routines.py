import torch
from torch import nn, optim
import data_generator
import numpy as np
from pathlib import Path
import os

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
    ninp = 50
    width = int(width)
    depth = int(depth)

    layer_sizes = [ninp]
    for i in range(depth - 1):
        layer_sizes.append((width))
    layer_sizes.append(1)
    model = get_neural_network(layer_sizes, nn.Tanh)

    cwd = os.getcwd()
    data_path = cwd + '/../../../data/function_approximation/IPL'

    train_x = np.loadtxt(data_path+'/Out'+str(int(np.log2(N)))+'.txt')[:,:ninp]
    train_y = torch.from_numpy(data_generator.weighted_function(train_x).reshape(train_x.shape[0],1))
    train_x = torch.from_numpy(train_x)

    test_x = np.loadtxt(data_path+'/Out15.txt')[1:,:ninp]
    test_y = torch.from_numpy(data_generator.weighted_function(test_x).reshape(test_x.shape[0], 1))
    test_x = torch.from_numpy(test_x)

    objective = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=reg)
    epochs = max_epochs

    def test(x,y):
        model.eval()
        with torch.no_grad():
            output = model(x.float())
            loss = torch.sqrt(objective(output,y.float()))
        return loss.item()

    for e in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_x.float())
        loss = objective(output, train_y.float())
        loss.backward()
        optimizer.step()

    output = model(train_x.float())
    train_loss = torch.sqrt(objective(output,train_y.float())).item()
    test_loss = test(test_x, test_y)

    Path('new_results').mkdir(parents=True, exist_ok=True)
    with open('new_results/N_'+str(int(N))+'.txt','a') as file:
        file.write(str(train_loss)+ ' ' + str(test_loss) + ' ' + str(int(id))+'\n')