#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 18:50:48 2020

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
from collections import defaultdict
from torch.autograd import Variable

from Federated.Utils import settings



class DatasetGenerator():
    
    '''
    Function for the old linear tests. 
    '''
    
    def __init__(self, num_workers, num_rounds):
        self.num_workers = num_workers
        self.num_rounds = num_rounds
        self.model_drift = False
    
    
    def generate(self, dataset_size, multi_features=False, model_drift=False):
        
        self.model_drift = model_drift
        train_list_X, train_list_y, test_X, test_y = synthetic_dataset_creator(dataset_size, 
                                                                               self.num_workers, 
                                                                               self.num_rounds, 
                                                                               multi_features, 
                                                                               model_drift)
        
        return train_list_X, train_list_y, test_X, test_y




def splitDataset(dataset_X, num_workers, num_rounds):
    
    '''
    Function for the old linear tests. 
    '''
    
    # total number of sets to generate
    a = num_workers * num_rounds
    
    b = math.floor(len(dataset_X)/a)
    x = len(dataset_X) / b
    
        
    return np.array_split(dataset_X, x)


def synthetic_dataset_creator(dataset_size, num_workers, num_rounds, multi_features=False, model_drift=False):
    
    '''
    Function for the old synthetic tests. 
    '''
    
    train_list_X = [] 
    train_list_y = []
    test_X = []
    test_y = []
    
    if model_drift :
        
        after_drift=False
        
        for i in range(settings.drifts+1):
            tr_X, tr_y, t_X, t_y = generate_data(int(dataset_size/(settings.drifts+1)), 
                                                 num_workers, 
                                                 int(num_rounds/(settings.drifts+1)), 
                                                 after_drift, 
                                                 multi_features)
            
            for j in range(len(tr_X)):
                train_list_X.append(tr_X[j])
                train_list_y.append(tr_y[j])
            
            test_X.append(t_X)
            test_y.append(t_y)
            
            after_drift=True
            
        
    else:
        train_list_X, train_list_y, test_X, test_y = generate_data(dataset_size, 
                                                                   num_workers, 
                                                                   num_rounds, 
                                                                   multi_features)
        
    
    return train_list_X, train_list_y, test_X, test_y


def generate_data(dataset_size, num_workers, num_rounds, after_drift=False, multi_features=False):
    
    '''
    Function for the old linear tests. 
    '''
    
    if multi_features is True:
        
        dataset_X1 = np.random.uniform(low=settings.range_min, high=settings.range_max, size=(dataset_size,))
        dataset_X2 = np.random.uniform(low=settings.range_min, high=settings.range_max, size=(dataset_size,))
        dataset_X3= np.random.uniform(low=settings.range_min, high=settings.range_max, size=(dataset_size,))

        np.random.shuffle(dataset_X1)
        np.random.shuffle(dataset_X2)
        np.random.shuffle(dataset_X3)
        
        dataset_X = np.array([dataset_X1, dataset_X2, dataset_X3])
        
        
        if after_drift:
            if settings.function_type == 'linear':
                dataset_y = dataset_X1 * settings.mm[0] + dataset_X2 * settings.mm[1] + dataset_X3 * settings.mm[2] + settings.mm[3] + np.random.randn(dataset_X1.size) * math.sqrt(settings.v)
            else:
                dataset_y = settings.mm[0] * np.sin(dataset_X1*(settings.mm[1]*math.pi)+ dataset_X2*(settings.mm[2]*math.pi) + dataset_X3*(settings.mm[3]*math.pi)) + settings.mm[4] + np.random.randn(dataset_X1.size) * math.sqrt(settings.v)
                
        else:
            if settings.function_type == 'linear': 
                dataset_y = dataset_X1 * settings.m[0] + dataset_X2 * settings.m[1] + dataset_X3 * settings.m[2] + settings.m[3] + np.random.randn(dataset_X1.size) * math.sqrt(settings.v)
            else:
                dataset_y = settings.m[0] * np.sin(dataset_X1*(settings.m[1]*math.pi)+ dataset_X2*(settings.m[2]*math.pi) + dataset_X3*(settings.m[3]*math.pi)) +settings.m[4]+ np.random.randn(dataset_X1.size) * math.sqrt(settings.v)
            
        dataset_y = dataset_y.reshape(-1,1)
        dataset_X = dataset_X.transpose()
        
    else:
        
        dataset_X = np.random.uniform(low=settings.range_min, high=settings.range_max, size=(dataset_size,))
        np.random.shuffle(dataset_X)
        
        if after_drift:
            if settings.function_type == 'linear':
                dataset_y =  dataset_X * settings.mm[0] + settings.mm[1] +  np.random.randn(dataset_X.size) * math.sqrt(settings.v)
            else:
                dataset_y = settings.mm[0] * np.cos(dataset_X*(settings.mm[1]*math.pi)) + settings.mm[2] + np.random.randn(dataset_X.size) * math.sqrt(settings.v)
        else:
            if settings.function_type == 'linear':
                dataset_y =  dataset_X * settings.m[0] + settings.m[1] +  np.random.randn(dataset_X.size) * math.sqrt(settings.v)
            else:
                dataset_y =settings.m[0] * np.sin(dataset_X*(settings.m[1]*math.pi)) + settings.m[2] + np.random.randn(dataset_X.size) * math.sqrt(settings.v)

        dataset_X = dataset_X.reshape(-1,1)
        dataset_y = dataset_y.reshape(-1,1)
    
    
    train_X, test_X = np.split(dataset_X, 
                [int(settings.train_percentage * len(dataset_X))
                ])

    train_y, test_y = np.split(dataset_y, 
                [int(settings.train_percentage * len(dataset_y))
                ])
    
    
    train_list_X = splitDataset(train_X, num_workers, num_rounds)
    train_list_y = splitDataset(train_y, num_workers, num_rounds)
    

    for i in range(0, len(train_list_X)):
        train_list_X[i] = torch.from_numpy(train_list_X[i])

    for i in range(0, len(train_list_y)):
        train_list_y[i] = torch.from_numpy(train_list_y[i])
    
    
    return train_list_X, train_list_y, test_X, test_y


def trainFiltering(dataset, remove_list, num_clients):
    '''
        - Remove data with target in remove_list
        - Select the correct subset of dataset in order to perform splitting among all clients
    '''
    for elem in remove_list:
        idx = dataset.targets != elem
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]
    
    len_train_dataset = int(dataset.data.shape[0] / num_clients) * num_clients
    
    dataset.data = dataset.data[:len_train_dataset]
    dataset.targets = dataset.targets[:len_train_dataset]
    
    return dataset


def testFiltering(dataset, remove_list, len_dataset):
    '''
        - Remove data with target in remove_list
        - Set the dimension of the test dataset
    '''
    for elem in remove_list:
        idx = dataset.targets != elem
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]
    
    dataset.data = dataset.data[:len_dataset]
    dataset.targets = dataset.targets[:len_dataset]
    
    return dataset