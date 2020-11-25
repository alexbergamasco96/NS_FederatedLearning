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
num_rounds = 40

dataset_size = 6000    #dataset size
multifeatures = False
model_drift = True


learning_rate = 1e-3
local_epochs = 300

lr_gamma_FedREG = 1
lr_gamma_FedAVG = 0.8

lr_decay = True





def set_new_seed(x):
    
    np.random.seed(x)
    torch.manual_seed(x)

    
def single_iteration(seed):
    
    #setting the current seed
    set_new_seed(seed)
    
    ### Dataset Creation
    
    train_list_X, train_list_y, test_X, test_y = synthetic_dataset_creator(dataset_size, num_workers, num_rounds, multi_features=multifeatures, model_drift=model_drift)
    
    
    w, w_avg = model_creator(   input_size=len(train_list_X[0][0]), 
                                output_size=len(train_list_y[0][0]), 
                                num_workers=num_workers,
                                hidden=64,
                                model_type='periodic'
                            )

    criterion, optimizers = loss_optimizer(models=w, 
                                           learning_rate=learning_rate, 
                                           gamma=lr_gamma_FedREG, 
                                           decay=False, 
                                           local_epochs=local_epochs)
    
        
    params = get_models_parameters(models_list=w)
        
        
    for model in w:
        model.train()

    error = []
    score = []


    ### OUR ALGORITHM - EXECUTION
    
    # Parameter 'c' of FedREG algorithm
    c = 0
    
    global_params = calculate_FedAVG_params(w, params)
    
    
    # EXECUTION
    for i in range(0, num_rounds):
        
        # Two different test sets, one for each dataset:
        # before drift and after drift
        
        if model_drift:
            if i < (num_rounds/2):
                current_test_X = test_X[0]
                current_test_y = test_y[0]
            else:
                current_test_X = test_X[1]
                current_test_y = test_y[1]
        else:
            current_test_X = test_X[0]
            current_test_y = test_y[0]
            
            
        for j in range(0, num_workers):
            trainInBatch(  model=w[j],
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
            global_params, beta = calculate_FedREG_params_with_adaption( models=w, 
                                                                   global_params=global_params, 
                                                                   new_params=new_params, 
                                                                   current_round=i,
                                                                   c=c)
            
            # Set new aggregated parameters
            set_parameters(global_params, w)
        
        
            # Perform the prediction
            #predicted = w[0](Variable(torch.from_numpy(current_test_X).float())).data.numpy()
            predicted = w[0](torch.from_numpy(current_test_X).float()).data.numpy()
            
            
            error.append(mean_squared_error(current_test_y, predicted))
            score.append(r2_score(current_test_y, predicted))
        
        f = open("log_file.txt", "a")
        f.write("PARAMETER C: {}\n".format(c))
        f.close()
        
        # Update parameter C
        c = c*beta + len(w)
        

        
    ### -----------------------------------------------------------------------
    ### -----------------------------------------------------------------------
    ### -----------------------------------------------------------------------
    ### -----------------------------------------------------------------------
    
    

    
    ### FEDAVG - INITIALIZATION
    
    criterion_avg, optimizers_avg = loss_optimizer(models=w_avg, 
                                                   learning_rate=learning_rate, 
                                                   gamma=lr_gamma_FedAVG, 
                                                   decay=False, 
                                                   local_epochs=(local_epochs-10))
    
        
    params = get_models_parameters(w_avg)
        
        
    for model in w_avg:
        model.train()
    
    
    error_fedavg = []
    score_fedavg = []

    
    train_list_X = copy.deepcopy(train_list_X)
    train_list_y = copy.deepcopy(train_list_y)
    test_X = copy.deepcopy(test_X)
    test_y = copy.deepcopy(test_y)
    
    ### FEDAVG - EXECUTION
    
    for i in range(0, num_rounds):
        
        if model_drift:
            if i < (num_rounds/2):
                current_test_X = test_X[0]
                current_test_y = test_y[0]
            else:
                current_test_X = test_X[1]
                current_test_y = test_y[1]
        else:
            current_test_X = test_X[0]
            current_test_y = test_y[0]
            
        
        for j in range(0, num_workers):
            trainInBatch(  model=w_avg[j],
                    optimizer=optimizers_avg[j],
                    criterion=criterion_avg[j],
                    inputs=train_list_X[i*num_workers+j],
                    labels=train_list_y[i*num_workers+j],
                    local_epochs=local_epochs,
                    decay=False,
                    input_len = len(train_list_X[i*num_workers+j]))
            
            if lr_decay:
                with torch.no_grad():
                    # Learning Rate Decay
                    for g in optimizers_avg[j].param_groups:
                        g['lr'] = learning_rate / (1 + (i+1))

            
        # Get the params
        params = get_models_parameters(w_avg)
        
       
        with torch.no_grad():

            new_params = calculate_FedAVG_params(w_avg, params)

            set_parameters(new_params, w_avg)
            
            
            #predicted = w_avg[0](Variable(torch.from_numpy(current_test_X).float())).data.numpy()
            predicted = w_avg[0](torch.from_numpy(current_test_X).float()).data.numpy()
            
            error_fedavg.append(mean_squared_error(current_test_y, predicted))
            score_fedavg.append(r2_score(current_test_y, predicted))
        
        for g in optimizers_avg[j].param_groups:
            f = open("log_file.txt", "a")
            f.write("LR {}\n".format(g['lr']))
            f.close()
    
    return error, score, error_fedavg, score_fedavg