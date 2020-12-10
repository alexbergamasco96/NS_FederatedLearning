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
    
    
    
    # --- FedAVG
    set_new_seed(seed)
    
    server_fedavg = Server(num_workers=settings.num_workers, 
                           model_type=settings.model_type, 
                           aggregation_method='FedAVG')
    
    server_fedavg.generateDataset(dataset_size=settings.dataset_size, 
                                  num_rounds=settings.num_rounds, 
                                  multi_features=settings.multi_features, 
                                  model_drift=settings.model_drift)
    
    error_list_fedavg, score_list_fedavg = server_fedavg.fullTraining()
    
    
    
    
    # --- FedREG
    set_new_seed(seed)
    
    server_fedreg = Server(num_workers=settings.num_workers, 
                           model_type=settings.model_type, 
                           aggregation_method='FedREG')
    
    server_fedreg.generateDataset(dataset_size=settings.dataset_size, 
                                  num_rounds=settings.num_rounds, 
                                  multi_features=settings.multi_features, 
                                  model_drift=settings.model_drift)
    
    error_list_fedreg, score_list_fedreg = server_fedreg.fullTraining()
    
    
    
    
    # --- FedREG_Distance
    set_new_seed(seed)
    
    server_fedreg_distance = Server(num_workers=settings.num_workers, 
                                    model_type=settings.model_type, 
                                    aggregation_method='FedREG_Distance')
    
    server_fedreg_distance.generateDataset(dataset_size=settings.dataset_size, 
                                           num_rounds=settings.num_rounds, 
                                           multi_features=settings.multi_features, 
                                           model_drift=settings.model_drift)
    
    error_list_fedreg_distance, score_list_fedreg_distance = server_fedreg_distance.fullTraining()
    
    
    return error_list_fedavg, score_list_fedavg, error_list_fedreg, score_list_fedreg, error_list_fedreg_distance, score_list_fedreg_distance