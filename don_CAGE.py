#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 03:04:08 2024

@author: saumya
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
###############################################################################
from gstools import SRF, Gaussian
from gstools.random import MasterRNG
###############################################################################
from scipy import integrate
###############################################################################
m1 = 202
m2 = 303
n = 4
###############################################################################
x = np.linspace(0, 1, m2)
###############################################################################
import scipy.io as sio
import os




#cwd = os.getcwd()
#os.chdir(cwd)
path = '/home/saumya/Desktop/q control/spyder tries'
os.environ['PATH'] += ':'+path


mat = sio.loadmat('ThrLev.mat')    #loading data sets with prob >= 0.99 and not noisy
V = mat['omTrain']
Ef = mat['FocTrain']
###############################################################################
def integrate_data(m1=m1, m2=m2, n=n, x=x, U=V, S=Ef):
    us = np.zeros((m2*n, m1))
    xs = np.zeros((m2*n, 1))
    ss = np.zeros((m2*n, 1))
    for i in range(n):
        for j in range(m2):
            us[i*m2+j, :] = V[i, :]
            xs[i*m2+j, :] = x[j]
            ss[i*m2+j, :] = Ef[j, i]
    return us, xs, ss
###############################################################################
us, xs, ss = integrate_data()
###############################################################################
def batch_dataset(batch_size, m1=m1, m2=m2, n=n, x=x, U=V, S=Ef, ratio=0.9):
    us, xs, ss = integrate_data(m1, m2, n, x, U=V, S=Ef)
    train_size = int(len(us)*ratio)
    us_train = us[:train_size]
    xs_train = xs[:train_size]
    ss_train = ss[:train_size]

    us_test = us[train_size:]
    xs_test = xs[train_size:]
    ss_test = ss[train_size:]

    us_train = torch.tensor(us_train, dtype=torch.float32)
    xs_train = torch.tensor(xs_train, dtype=torch.float32)
    ss_train = torch.tensor(ss_train, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(us_train, xs_train, ss_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    us_test = torch.tensor(us_test, dtype=torch.float32)
    xs_test = torch.tensor(xs_test, dtype=torch.float32)
    ss_test = torch.tensor(ss_test, dtype=torch.float32)
    test_dataset = torch.utils.data.TensorDataset(us_test, xs_test, ss_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader
###############################################################################
train_data, test_data = batch_dataset(batch_size=32, ratio=0.9)
###############################################################################
class DeepONet(nn.Module):
    def __init__(self, neurons=40, in1=1, in2=1, output_neurons=20):
        super(DeepONet, self).__init__()
        self.in1 = in1
        self.in2 = in2
        self.output_neurons = output_neurons
        self.neurons = neurons

        self.branch = self.branch_network()
        self.trunk = self.trunk_network()

    def branch_network(self):
        branch = nn.Sequential(nn.Linear(self.in1, self.neurons), nn.ReLU(),
                               nn.Linear(self.neurons, self.neurons), nn.ReLU(),
                               nn.Linear(self.neurons, self.output_neurons))
        return branch


    def trunk_network(self):
        trunk = nn.Sequential(nn.Linear(self.in2, self.neurons), nn.ReLU(),
                               nn.Linear(self.neurons, self.neurons), nn.ReLU(),
                               nn.Linear(self.neurons, self.neurons), nn.ReLU(),
                               nn.Linear(self.neurons, self.output_neurons))
        return trunk


    def forward(self, x1, x2):
        x1 = self.branch(x1)
        x2 = self.trunk(x2)
        x = torch.einsum("bi, bi-> b", x1, x2)
        x = torch.unsqueeze(x, 1)
        return x
###############################################################################
model = DeepONet(neurons = 40, in1 = m1, in2=1)
###############################################################################
def loss(y_pred, y):
    return torch.mean((y_pred - y)**2)
###############################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
###############################################################################
train_losses = []
#epochs = 2000
epochs = 500
for i in range(epochs):
    l_total = 0
    for u_batch, x_batch, s_batch in train_data:
        model.train()
        optimizer.zero_grad()
        y_pred = model(u_batch, x_batch)

        l = loss(y_pred, s_batch)
        l_total += l.item()
        l.backward()
        optimizer.step()

    l_total = l_total/len(train_data)
    train_losses.append(l_total)

#    print("i: ", i)
    if i%10 == 0:
        print("Epoch:  ", i, "/", epochs, "loss: ", l_total)
###############################################################################
u_t = V[1200,:]
s_t = Ef[1200,:]
u_t_tens = torch.tensor(u_t, dtype = torch.float32)
#print(u_t_tens)
u_t_tens = torch.tensor(u_t, dtype = torch.float32).unsqueeze(0)
#print(u_t_tens)
x_tens = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
#print(x_tens)
###############################################################################
model.eval()
p = model(u_t_tens, x_tens)
p=p.detach().numpy()
###############################################################################
plt.plot(x, s_t, label = "integration analytical")
plt.plot(x, p, label = "integration DNN model")
plt.legend()
###############################################################################


