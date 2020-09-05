import torch
from torch import nn
import numpy as np
import math
from matplotlib import pyplot as plt
import os

seed = 0
torch.manual_seed(seed)

def HO_loss(x1, y1, x2, y2):
    loss = nn.MSELoss()
    return 2.*loss(x1, y1) - loss(x2, y2)

def test_LOSS(x1, y1, x2, y2):
    loss = nn.MSELoss()
    return torch.sqrt(2.*loss(x1, y1) - loss(x2, y2))

def get_neural_network(layer_sizes, activationFunction):
    model = nn.Sequential()
    model.add_module('In', nn.Linear(layer_sizes[0], layer_sizes[1]))
    model.add_module('sigma_1', activationFunction())
    for i in range(2, len(layer_sizes) - 1):
        model.add_module('Transformation_' + str(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        model.add_module('sigma_' + str(i), activationFunction())
    model.add_module('Transformation_' + str(len(layer_sizes) - 1), nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    return model

def clip_weight(model, clip):
    for name, p in model.named_parameters():
        if ('In' in name and 'weight' in name):
            for i in range(p.data.size(1)):
                p.data[:, i] = p.data[:, i].clamp_(-clip[i], clip[i])
        if ('Transformation_' in name and 'weight' in name):
            p.data = p.data.clamp_(-math.pi/4., math.pi/4.)

start = 3
end = 11

ninp = 16
width = 5
depth = 2
clip = torch.arange(1,ninp+1).float()**(-2.5)

layer_sizes = [ninp]
for i in range(depth - 1):
    layer_sizes.append((width))
layer_sizes.append(1)
model = get_neural_network(layer_sizes, nn.Tanh)


train_errors_non_clipped = []
train_errors_clipped = []

cwd = os.getcwd()
data_path = cwd + '/../../../data/elliptic/standard_data'

test_x1 = torch.from_numpy(np.loadtxt(data_path+'/dim_'+str(ninp)+'/In_12.txt',delimiter=',')[:,1:].T)
test_y1 = torch.from_numpy(np.loadtxt(data_path+'/dim_'+str(ninp)+'/Out12.txt',skiprows=1)[1:,1]).unsqueeze(-1)

test_x2 = torch.from_numpy(np.loadtxt(data_path+'/dim_'+str(ninp)+'/In_11.txt',delimiter=',')[:,1:].T)
test_y2 = torch.from_numpy(np.loadtxt(data_path+'/dim_'+str(ninp)+'/Out11.txt',skiprows=1)[1:,1]).unsqueeze(-1)

## run the non-holomorphic case
for i, N in enumerate(range(start,end)):
    train_x1 = torch.from_numpy(np.loadtxt(data_path + '/dim_' + str(ninp) + '/In_' + str(int(N)) + '.txt', delimiter=',').T)
    train_y1 = torch.from_numpy(np.loadtxt(data_path + '/dim_' + str(ninp) + '/Out' + str(int(N)) + '.txt', skiprows=1)[:,1]).unsqueeze(-1)

    train_x2 = torch.from_numpy(np.loadtxt(data_path + '/dim_' + str(ninp) + '/In_' + str(int(N - 1)) + '.txt', delimiter=',').T)
    train_y2 = torch.from_numpy(np.loadtxt(data_path + '/dim_' + str(ninp) + '/Out' + str(int(N - 1)) + '.txt', skiprows=1)[:,1]).unsqueeze(-1)

    output1 = model(train_x1.float())
    output2 = model(train_x2.float())
    trainloss = torch.sqrt(HO_loss(output1, train_y1.float(), output2, train_y2.float()))
    train_errors_non_clipped.append(trainloss.item())

output_test1 = model(test_x1.float())
output_test2 = model(test_x2.float())
testloss = test_LOSS(output_test1, test_y1.float(),output_test2,test_y2.float()).item()
errors_non_clipped = np.abs(np.array(train_errors_non_clipped)-testloss)

## run the holomorphic case:
clip_weight(model, clip)
for i, N in enumerate(range(start,end)):
    train_x1 = torch.from_numpy(np.loadtxt(data_path + '/dim_' + str(ninp) + '/In_' + str(int(N)) + '.txt', delimiter=',').T)
    train_y1 = torch.from_numpy(np.loadtxt(data_path + '/dim_' + str(ninp) + '/Out' + str(int(N)) + '.txt', skiprows=1)[:,1]).unsqueeze(-1)

    train_x2 = torch.from_numpy(np.loadtxt(data_path + '/dim_' + str(ninp) + '/In_' + str(int(N - 1)) + '.txt', delimiter=',').T)
    train_y2 = torch.from_numpy(np.loadtxt(data_path + '/dim_' + str(ninp) + '/Out' + str(int(N - 1)) + '.txt', skiprows=1)[:,1]).unsqueeze(-1)

    output1 = model(train_x1.float())
    output2 = model(train_x2.float())
    trainloss = torch.sqrt(HO_loss(output1, train_y1.float(), output2, train_y2.float()))
    train_errors_clipped.append(trainloss.item())

output_test1 = model(test_x1.float())
output_test2 = model(test_x2.float())
testloss = test_LOSS(output_test1, test_y1.float(),output_test2,test_y2.float()).item()
errors_clipped = np.abs(np.array(train_errors_clipped)-testloss)

Ns = 2**np.arange(start,end)
points = Ns+ Ns/2.

B = np.vstack([np.log(points), np.ones_like(Ns)]).T
m_gen1, c_gen1 = np.linalg.lstsq(B, np.log(errors_non_clipped),rcond=None)[0]
print("decay rate gap non-holomorphic case: ", -m_gen1)

m_gen2, c_gen2 = np.linalg.lstsq(B, np.log(errors_clipped),rcond=None)[0]
print("decay rate gap holomorphic case: ", -m_gen2)


plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()

ax.loglog(points, errors_clipped, 'blue', linestyle="", marker="o",
          label=r'$ (\beta,p,\epsilon)$-holomorphic (rate: ' + str(round(-m_gen2, 1)) + ')', basex=2, markersize=8, zorder=10)
ax.loglog(points, errors_non_clipped, 'black', linestyle="", marker="s", label=r'Non-$ (\beta,p,\epsilon)$-holomorphic (rate: '+ str(round(-m_gen1, 1)) + ')', basex=10, markersize=8,
          zorder=10)

ax.loglog(points, np.exp(m_gen1 * np.log(points) + c_gen1), 'black', linewidth=3)
ax.loglog(points, np.exp(m_gen2 * np.log(points) + c_gen2), 'blue', linewidth=3)

plt.ylabel(r'$|\mathcal{E}_G -\mathcal{E}_T|$')
plt.xlabel(r'\# sample points $N$')
plt.grid(b=True, which='both', linestyle='--')
plt.legend()
plt.show()