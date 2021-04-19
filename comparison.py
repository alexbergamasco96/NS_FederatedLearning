#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 09:37:46 2020

@author: alex
"""

import torch
import numpy as np


from Federated.Server.server import Server
from Federated.Utils import settings


def set_new_seed(x):
    '''
    Set the seed for the current computation
    '''
    np.random.seed(x)
    torch.manual_seed(x)
    
    
    
def comparison(seed):
    '''
     Compare Two different algorithms on the same data and same initialization of NN Params
    '''
    # Set number of max thread not to use all the CPU resources
    torch.set_num_threads(2)
    
    # Model Type (MNISTFFNN, MNISTCNN, CIFARCNN)
    model_type = 'MNISTFFNN'
    
    # --- FedAdapt
    
    # Important to have the same weight initialization
    set_new_seed(seed)
    
    server_fedadapt = Server(num_workers=settings.num_workers, 
                           model_type=model_type, 
                           aggregation_method='FedAda', # FedAda: Adaptive-FedAVG
                           optimizer='SGD',
                           LRdecay=True)
    
    # Generate the dataset (choose the correct model_type)
    server_fedadapt.generateMNISTDataset(num_rounds=settings.num_rounds) 
    
    # Multi-Round Computation
    error_list_fedadapt, score_list_fedadapt = server_fedadapt.fullTrainingMNIST()
    
    
    
    
    # --- FedAVG
    
    # Starting from the same weights
    set_new_seed(seed)
    
    
    server_fedavg = Server(num_workers=settings.num_workers, 
                           model_type=model_type, 
                           aggregation_method='FedAVG',
                           optimizer='SGD', 
                           LRdecay=True)
    
    # Generate the dataset (choose the correct model_type)
    server_fedavg.generateMNISTDataset(num_rounds=settings.num_rounds)
    
    # Multi-Round Computation
    error_list_fedavg, score_list_fedavg = server_fedavg.fullTrainingMNIST()
    
    return error_list_fedadapt, score_list_fedadapt, error_list_fedavg, score_list_fedavg 