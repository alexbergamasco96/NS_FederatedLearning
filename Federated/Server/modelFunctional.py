#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:24:38 2020

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





def setLossOptimizer(model, learning_rate, optimizer):        
    criterion = torch.nn.MSELoss()
    
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   
        
    return criterion, optimizer


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
                global_params[i][j] = ((c * global_params[i][j] + new_params[i][j]) / (c + num_clients)).data.detach().clone()
    
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