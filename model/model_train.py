import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import numpy as np
import math

from Stationary.core import *
from Stationary.utils import *

from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

import torch.nn.functional as F
import copy

from collections import defaultdict

from torch.autograd import Variable




def loss_optimizer(models, learning_rate, gamma, local_epochs, decay=True):
    
    optimizers = []
    criterion = []
    
    for i in models:
        criterion.append(torch.nn.MSELoss()) 
        
        if decay is True:
            optimizers.append(torch.optim.lr_scheduler.StepLR(torch.optim.SGD(i.parameters(), lr=learning_rate),
                                                          step_size = local_epochs,
                                                          gamma=gamma))
        else:    
            optimizers.append(torch.optim.SGD(i.parameters(), lr=learning_rate))   
    
    return criterion, optimizers




def get_models_parameters(models_list):
    
    params = []
    
    for i in models_list:
        params.append(list(i.parameters()))
    
    return params
    

    
    
    
def train(model, criterion, optimizer, inputs, labels, local_epochs, decay, input_len):
    
    for epoch in range(local_epochs):
        
        if decay is True:
            optimizer.optimizer.zero_grad()
        else:
            optimizer.zero_grad()
        
        loss = 0

        for x in range(input_len):
            input_ = (inputs[x]).float()
            input_ = input_.unsqueeze(0)
            label = (labels[x]).float()
            label = label.unsqueeze(0)
            y_pred = model(input_)
            loss += criterion(y_pred, label)

        loss.backward()
        
        if decay is True:
            optimizer.optimizer.step()
        else:
            optimizer.step()
        
    
    
    
    
def sum_of_params(models, params):
    
    new_params = []
    
    for param_i in range(len(params[0])):
        spdz_params = list()

        for remote_index in range(len(models)):
            spdz_params.append(params[remote_index][param_i])

        spdz = torch.tensor([0.0]).float()

        for k in spdz_params:
            spdz = spdz + k

        new_param = (spdz)
        new_params.append(new_param)
    
    return new_params        
        
    
    
    
        
def calculate_FedREG_params(models, global_params, new_params, c):
    
    with torch.no_grad():
        for i in range(len(global_params)):
            for j in range(len(global_params[i])):
                global_params[i][j] = ((c * global_params[i][j] + new_params[i][j]) / (c + len(models))).data.detach().clone()
    
    return global_params
    
    
    
def calculate_FedREG_params_with_adaption(global_params, new_params, current_round, c):
    '''
        global_params:  last aggregated parameters
        new_params:     average of parameters received at this round
        round:          current round of computation 
    '''
    
    with torch.no_grad():
        
        # calculating the distance from the new parameters
        distance = 0
        for i in range(len(global_params)):
            distance += torch.norm((global_params[i] - new_params[i])/ global_params[i]) / len(global_params[i].view(-1,1))
        
        # distance is divided by the len of params
        distance /= len(global_params)

        # setting beta: parameter that express the distance 
        if current_round == 0:
            # first round we don't care about the distance. First iteration
            beta = 1
        else:
            # If distance = inf, then beta is 0 and the algorithm becomes FedAVG
            beta = 1 / (1 + distance) 

        for i in range(len(global_params)):
            for j in range(len(global_params[i])):
                global_params[i][j] = ((c *beta* global_params[i][j] + new_params[i][j]) / (c*beta + len(w))).data.detach().clone()
    
    return global_params
    
    
    

def calculate_FedAVG_params(models, params):
    
    with torch.no_grad():
        new_params = []

        for param_i in range(len(params[0])):

            spdz_params = []

            for remote_index in range(len(models)):
                spdz_params.append(params[remote_index][param_i])

            spdz = torch.tensor([0.0]).float()

            for k in spdz_params:
                spdz = spdz + k

            new_param = (spdz) / len(models)
            new_params.append(new_param)
    
    return new_params






def set_parameters(new, models):
    
    with torch.no_grad():
        for remote_index in range(len(models)):
            param_index = 0

            for p in models[remote_index].parameters():
                p.data = new[param_index].data.detach().clone()
                param_index += 1