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



''' SETTING THE STANDARD SEED '''
np.random.seed(0)
torch.manual_seed(0)



################
################



#----- Federated Parameters
num_workers = 4        # Number of workers
num_rounds = 40        # Total number of rounds

dataset_size = 6000    # dataset size
multifeatures = False  # Multi Feature inputs
model_drift = True     # Drift of the model after [num_rounds/2] rounds


learning_rate = 1e-3   # LR (same for FedAVG and FedREG)
local_epochs = 300     # Epochs performed client-side
batch_size = 8         # Batch_size for local training phase

lr_decay = True        # False (= decay not required) in case of Full Gradient Descent


# Parameter not used. Decay is implemented without scheduler, following "On the convergence of FedAVG on Non-IID Data"
lr_gamma_FedREG = 1
lr_gamma_FedAVG = 1







def set_new_seed(x):
    '''
    Set the seed for the current computation
    '''
    np.random.seed(x)
    torch.manual_seed(x)

    
def single_iteration(seed):
    
    # setting the current seed
    set_new_seed(seed)
    
    
    ### DATASET Creation
    
    train_list_X, train_list_y, test_X, test_y = synthetic_dataset_creator(dataset_size, 
                                                                           num_workers, 
                                                                           num_rounds, 
                                                                           multi_features=multifeatures, 
                                                                           model_drift=model_drift)
    
    # Instantiate all the models with the same weights
    w, w_avg = model_creator(input_size=len(train_list_X[0][0]), 
                             output_size=len(train_list_y[0][0]), 
                             num_workers=num_workers,
                             hidden=64,
                             model_type='periodic')
    
    # Setting loss and optimizer 
    criterion, optimizers = loss_optimizer(models=w, 
                                           learning_rate=learning_rate, 
                                           gamma=lr_gamma_FedREG, 
                                           local_epochs=local_epochs)
    
        
    params = get_models_parameters(models_list=w)
        
        
    for model in w:
        model.train()

    error = []
    score = []


    ### OUR ALGORITHM - EXECUTION
    
    # Parameter 'C' of FedREG algorithm
    # It defines the relevance of History for the new update
    c = 0
    
    # Calculating the global parameters for the first execution. A simple Average
    global_params = calculate_FedAVG_params(w, params)
    
    torch.manual_seed(seed)
    # EXECUTION
    for i in range(0, num_rounds):
        
        # Two different test sets, one for each dataset:
        # BEFORE drift and AFTER drift
        
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
                    batch_size=batch_size,
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
            
            # Appending the error and the score
            error.append(mean_squared_error(current_test_y, predicted))
            score.append(r2_score(current_test_y, predicted))
        
        
        ''' ---DEBUG--- Current parameter C
        f = open("log_file.txt", "a")
        f.write("PARAMETER C: {}\n".format(c))
        f.close()
        '''
        
        # Update parameter C
        c = c*beta + len(w)
        

        
    ### -----------------------------------------------------------------------
    ### -----------------------------------------------------------------------
    ### -----------------------------------------------------------------------
    ### -----------------------------------------------------------------------
    
    set_new_seed(seed)
    ### FEDAVG - INITIALIZATION
    
    # Setting loss and optimizer 
    criterion_avg, optimizers_avg = loss_optimizer(models=w_avg, 
                                                   learning_rate=learning_rate, 
                                                   gamma=lr_gamma_FedAVG, 
                                                   local_epochs=local_epochs)
    
        
    params = get_models_parameters(w_avg)
        
        
    for model in w_avg:
        model.train()
    
    
    error_fedavg = []
    score_fedavg = []

    
    '''
    train_list_X = copy.deepcopy(train_list_X)
    train_list_y = copy.deepcopy(train_list_y)
    test_X = copy.deepcopy(test_X)
    test_y = copy.deepcopy(test_y)
    '''
    
    
    ### FEDAVG - EXECUTION
    torch.manual_seed(seed)
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
                    batch_size=batch_size,
                    decay=False,
                    input_len = len(train_list_X[i*num_workers+j]))
            
            # Learning Rate Decay (needed for convergence): lr_t = (lr0)/(1+t) : t=current_round
            if lr_decay:
                with torch.no_grad():
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
        
    
    return error, score, error_fedavg, score_fedavg