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
    
    model_type = 'MNISTCNN'
    
    # --- FedAdapt
    set_new_seed(seed)
    
    server_fedadapt = Server(num_workers=settings.num_workers, 
                           model_type=model_type, 
                           aggregation_method='FedAda',
                           optimizer='SGD',
                           LRdecay=True)
    
    server_fedadapt.generateMNISTDataset(num_rounds=settings.num_rounds)
    
    error_list_fedadapt, score_list_fedadapt = server_fedadapt.fullTrainingMNIST()
    
    
    
    
    # --- FedAVG
    set_new_seed(seed)
    
    server_fedavg = Server(num_workers=settings.num_workers, 
                           model_type=model_type, 
                           aggregation_method='FedAVG',
                           optimizer='SGD',
                           LRdecay=True)
    
    server_fedavg.generateMNISTDataset(num_rounds=settings.num_rounds)
    
    error_list_fedavg, score_list_fedavg = server_fedavg.fullTrainingMNIST()
    
    return error_list_fedadapt, score_list_fedadapt, error_list_fedavg, score_list_fedavg 