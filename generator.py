import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import numpy as np
import math

from Stationary.core import *
from utils.dataset_utils import *
from model.model_creation import *
from model.model_train import *

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
num_rounds = 10

dataset_size = 500    #dataset size
multifeatures = False


learning_rate = 1e-3
local_epochs = 100

lr_gamma_FedREG = 1
lr_gamma_FedAVG = 0.9






def set_new_seed(x):
    
    np.random.seed(x)
    torch.manual_seed(x)

    
def single_iteration(seed):
    
    #setting the current seed
    set_new_seed(seed)
    
    ### Dataset Creation
    
    train_list_X, train_list_y, test_X, test_y = synthetic_dataset_creator(dataset_size, num_workers, num_rounds, multi_features=multifeatures)
    
    
    w, w_avg = model_creator(   input_size=len(train_list_X[0][0]), 
                                output_size=len(train_list_y[0][0]), 
                                num_workers=num_workers,
                                hidden=64
                            )

    criterion, optimizers = loss_optimizer(models=w, learning_rate=learning_rate, gamma=lr_gamma_FedREG, decay=False, local_epochs=local_epochs)
    
        
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
    
    criterion_avg, optimizers_avg = loss_optimizer(models=w_avg, learning_rate=learning_rate, gamma=lr_gamma_FedAVG, decay=True, local_epochs=local_epochs)
    
        
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