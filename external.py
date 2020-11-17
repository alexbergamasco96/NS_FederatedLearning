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





np.random.seed(0)
torch.manual_seed(0)



################
################


#-----Federated Parameters
num_workers = 4
num_rounds = 30


#-----Linear Regression Parameters
m = -2.1
m2 = 1.4
m3 = -0.5
c = 1.4
a = 0.1
v = 0.3    #noise variance
range_min = 0    #min value of X
range_max = 20    #max value of X
dataset_size = 4500    #dataset size



#-----FedAVG Parameters
learning_rate = 1e-3
local_epochs = 300
lr_gamma_FedREG = 1
lr_gamma_FedAVG = 0.8

#-----Execution Parameters
iterations = 20
train_percentage = 0.8





def set_new_seed(x):
    
    np.random.seed(x)
    torch.manual_seed(x)
    
    
    
    

class customModel(torch.nn.Module):
    
    def __init__(self, inputSize, outputSize, H = 64):
        
        super(customModel, self).__init__()
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

    
    


def synthetic_dataset_creator(multi_features=False):
    
    if multi_features is True:
        
        dataset_X1 = np.random.uniform(low=range_min, high=range_max, size=(dataset_size,))
        dataset_X2 = np.random.uniform(low=range_min, high=range_max, size=(dataset_size,))
        dataset_X3= np.random.uniform(low=range_min, high=range_max, size=(dataset_size,))

        np.random.shuffle(dataset_X1)
        np.random.shuffle(dataset_X2)
        np.random.shuffle(dataset_X3)

        dataset_X = np.array([dataset_X1, dataset_X2, dataset_X3])
        #dataset_y = dataset_X1 * m + dataset_X2 * m2 + dataset_X3 * m3 + c + np.random.randn(dataset_X1.size) * math.sqrt(v)
        dataset_y = m * np.sin(dataset_X1*(a*math.pi)+ dataset_X2*(a*math.pi) + dataset_X3*(a*math.pi)) + np.random.randn(dataset_X1.size) * math.sqrt(v)
        dataset_y = dataset_y.reshape(-1,1)
        dataset_X = dataset_X.transpose()
        
    else:
        
        dataset_X = np.random.uniform(low=range_min, high=range_max, size=(dataset_size,))
        np.random.shuffle(dataset_X)

        #dataset_y =  dataset_X * m + c +  np.random.randn(dataset_X.size) * math.sqrt(v)
        dataset_y = m * np.sin(dataset_X*(a*math.pi)) + np.random.randn(dataset_X.size) * math.sqrt(v)
        
        dataset_X = dataset_X.reshape(-1,1)
        dataset_y = dataset_y.reshape(-1,1)
    
    
    train_X, test_X = np.split(dataset_X, 
                [int(train_percentage * len(dataset_X))
                ])

    train_y, test_y = np.split(dataset_y, 
                [int(train_percentage * len(dataset_y))
                ])
    
    
    train_list_X = splitDataset(train_X, num_workers, num_rounds)
    train_list_y = splitDataset(train_y, num_workers, num_rounds)
    

    for i in range(0, len(train_list_X)):
        train_list_X[i] = torch.from_numpy(train_list_X[i])

    for i in range(0, len(train_list_y)):
        train_list_y[i] = torch.from_numpy(train_list_y[i])
    
    
    return train_list_X, train_list_y, test_X, test_y    
    
    
    
    

def model_creator(input_size, output_size, hidden=10):
    
    w = []
    w.append(customModel(input_size, output_size, H=hidden))
    for i in range(1, num_workers):
        w.append(copy.deepcopy(w[0]))
    
    w_avg = []
    for i in range(0, num_workers):
        w_avg.append(copy.deepcopy(w[0]))
        
    return w, w_avg





def loss_optimizer(models, gamma, decay=True):
    
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


    
    
    
    
def single_iteration(seed):
    
    #setting the current seed
    set_new_seed(seed)
    
    ### Dataset Creation
    
    train_list_X, train_list_y, test_X, test_y = synthetic_dataset_creator(multi_features=False)
    
    
    w, w_avg = model_creator(   input_size=len(train_list_X[0][0]), 
                                output_size=len(train_list_y[0][0]), 
                                hidden=64
                            )

    criterion, optimizers = loss_optimizer(models=w, gamma=lr_gamma_FedREG, decay=False)
    
        
    params = get_models_parameters(models_list=w)
        
        
    for model in w:
        model.train()


    # stores losses trend for each worker along epochs
    worker_losses_dict = defaultdict(list)

    error = []
    score = []


    ### OUR ALGORITHM - EXECUTION
    
    # Parameter 'c' of FedREG algorithm
    c = 0
    
    global_params = calculate_FedAVG_params(w, params)
    
    
    # EXECUTION
    for i in range(0, num_rounds):
        
        for j in range(0, num_workers):
            train(  model=w[j],
                    optimizer=optimizers[j],
                    criterion=criterion[j],
                    inputs=train_list_X[i*num_workers+j],
                    labels=train_list_y[i*num_workers+j],
                    local_epochs=local_epochs,
                    decay=False,
                    input_len=len(train_list_X[i*num_workers+j]))
        
            
        # Get the params
        params = get_models_parameters(w)
        
        
        # Parameter Aggregation
        with torch.no_grad():

            new_params = sum_of_params(w, params)
            
            # Calculate the aggregated parameters with our method
            global_params = calculate_FedREG_params(w, global_params, new_params, c)
            
            # Set new aggregated parameters
            set_parameters(global_params, w)
        
        
            # Perform the prediction
            predicted = w[0](Variable(torch.from_numpy(test_X).float())).data.numpy()

            error.append(mean_squared_error(test_y, predicted))
            score.append(r2_score(test_y, predicted))
        
        # Update parameter C
        c = c + len(w)
        


    ### -----------------------------------------------------------------------
    ### -----------------------------------------------------------------------
    ### -----------------------------------------------------------------------
    ### -----------------------------------------------------------------------
    
    

    
    ### FEDAVG - INITIALIZATION
    
    criterion_avg, optimizers_avg = loss_optimizer(models=w_avg, gamma=lr_gamma_FedAVG, decay=True)
    
        
    params = get_models_parameters(w_avg)
        
        
    for model in w_avg:
        model.train()
    
    # stores losses trend for each worker along epochs
    worker_losses_dict = defaultdict(list)
    
    
    error_fedavg = []
    score_fedavg = []

    
    train_list_X = copy.deepcopy(train_list_X)
    train_list_y = copy.deepcopy(train_list_y)
    test_X = copy.deepcopy(test_X)
    test_y = copy.deepcopy(test_y)
    
    ### FEDAVG - EXECUTION
    
    for i in range(0, num_rounds):
        
        for j in range(0, num_workers):
            train(  model=w_avg[j],
                    optimizer=optimizers_avg[j],
                    criterion=criterion_avg[j],
                    inputs=train_list_X[i*num_workers+j],
                    labels=train_list_y[i*num_workers+j],
                    local_epochs=local_epochs,
                    decay=True,
                    input_len = len(train_list_X[i*num_workers+j]))
        
            
        # Get the params
        params = get_models_parameters(w_avg)
        
       
        with torch.no_grad():

            new_params = calculate_FedAVG_params(w_avg, params)

            set_parameters(new_params, w_avg)

            predicted = w_avg[0](Variable(torch.from_numpy(test_X).float())).data.numpy()
            
            error_fedavg.append(mean_squared_error(test_y, predicted))
            score_fedavg.append(r2_score(test_y, predicted))
        
     
    
    return error, score, error_fedavg, score_fedavg