#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 21:40:22 2020

@author: alex
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy
from torch.autograd import Variable
from Federated.Server.modelFunctional import *


class periodicModel(torch.nn.Module):
    
    def __init__(self, inputSize, outputSize, H = 64):
        
        super(periodicModel, self).__init__()
        self.linear = torch.nn.Linear(inputSize, H, bias=True)
        self.linear2 = torch.nn.Linear(H, H, bias=True)
        self.linear3 = torch.nn.Linear(H, H, bias=True)
        self.linear4 = torch.nn.Linear(H, outputSize)

        
    def forward(self, x):
        x = torch.tanh(self.linear(x))
        x = torch.tanh(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        x = self.linear4(x)
        return x

    
class linearModel(torch.nn.Module):
    
    def __init__(self, inputSize, outputSize, H = 20):
        
        super(linearModel, self).__init__()
        self.linear = torch.nn.Linear(inputSize, H, bias=True)
        self.linear2 = torch.nn.Linear(H, outputSize)

        
    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(x)
        return x
    

class nonLinearModel(torch.nn.Module):
    
    def __init__(self, inputSize, outputSize, H = 20):
        
        super(nonLinearModel, self).__init__()
        self.linear = torch.nn.Linear(inputSize, H, bias=True)
        self.linear2 = torch.nn.Linear(H, outputSize)

        
    def forward(self, x):
        x = F.relu(self.linear(x))
        x = self.linear2(x)
        return x
    

def createModel(input_size, output_size, initialLR, hidden=64, model_type='periodic', optimizer='SGD'):
    
    if model_type == 'linear':
        model = linearModel(input_size, output_size, H=hidden)
    elif model_type == 'non_linear':
        model = nonLinearModel(input_size, output_size, H=hidden)
    else:
        model = periodicModel(input_size, output_size, H=hidden)
    
    criterion, optimizer = setLossOptimizer(model, initialLR, optimizer)
    return model, criterion, optimizer

'''
def setLossOptimizer(model, learning_rate):        
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)    
    return criterion, optimizer
'''

def model_creator(input_size, output_size, num_workers, hidden=64, model_type='periodic'):
    
    w = []
    
    if model_type == 'linear':
        w.append(linearModel(input_size, output_size, H=hidden))
    elif model_type == 'non_linear':
        w.append(nonLinearModel(input_size, output_size, H=hidden))
    else:
        w.append(periodicModel(input_size, output_size, H=hidden))
    
  
    for i in range(1, num_workers):
        w.append(copy.deepcopy(w[0]))
    
    w_avg = []
    for i in range(0, num_workers):
        w_avg.append(copy.deepcopy(w[0]))
        
    return w, w_avg

"""
def paramsSum(num_clients, params):
    '''
    new_params contains the list of params, calculated as the sum over all the models
    '''
    new_params = []
    
    for param_i in range(len(params[0])):
        spdz_params = list()

        for remote_index in range(num_clients):
            spdz_params.append(params[remote_index][param_i])

        spdz = torch.tensor([0.0]).float()

        for k in spdz_params:
            spdz = spdz + k

        new_param = (spdz)
        new_params.append(new_param)
    
    return new_params  


def calculateFedREGParams(num_clients, global_params, new_params, c):
    '''
    Standard calculation of FedREG
    '''
    
    with torch.no_grad():
        for i in range(len(global_params)):
            for j in range(len(global_params[i])):
                global_params[i][j] = ((c * global_params[i][j] + new_params[i][j]) / (c + len(num_clients))).data.detach().clone()
    
    return global_params


def calculateFedREGParamsWithAdaption(num_clients, global_params, new_params, current_round, c):
    '''
    FedREG with distance term.
        global_params:  last aggregated parameters
        new_params:     sum of parameters received at this round
        round:          current round of computation 
    '''
    
    with torch.no_grad():
        
        # calculating the distance from the new parameters
        distance = 0
        for i in range(len(global_params)):
            distance += torch.norm((global_params[i] - new_params[i]/num_clients))
        
        # distance is divided by the len of params
        #distance /= len(global_params)

        # setting beta: parameter that express the distance 
        if current_round == 0:
            # first round we don't care about the distance. First iteration
            beta = 1
        else:
            # If distance = inf, then beta is 0 and the algorithm becomes FedAVG (but with constant LR)
            beta = (1 / (1 + distance)) 
        
        
        for i in range(len(global_params)):
            for j in range(len(global_params[i])):
                global_params[i][j] = ((c *beta* global_params[i][j] + new_params[i][j]) / (c*beta + num_clients)).data.detach().clone()
    
    
    return global_params, beta


def calculateFedAVGParams(num_clients, params):
    '''
    Perform the averaging of the parameters
    '''
    with torch.no_grad():
        new_params = []

        for param_i in range(len(params[0])):

            spdz_params = []

            for remote_index in range(num_clients):
                spdz_params.append(params[remote_index][param_i])

            spdz = torch.tensor([0.0]).float()

            for k in spdz_params:
                spdz = spdz + k

            new_param = (spdz) / num_clients
            new_params.append(new_param)
    
    return new_params
    
    
"""