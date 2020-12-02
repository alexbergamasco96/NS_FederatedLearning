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





#----- Regression Parameters


'''
    The function is defined as:    
    
    y = m[0]*sin(X*m[1]*phi)+m[2]
'''

''' Params for Periodic Function'''
'''
m = [10.0, 0.1, 1.5] #before drift
mm = [6.0, 0.15, -3.0] #after drift
'''

''' Params for Linear Function '''
m = [3.0, 0.5, 1.5, 1.0] #before drift
mm = [4.0, 1.0, 0.5, 2.5] #after drift



v = 5 # noise

function_type = 'linear' # 'linear' or 'periodic'



drifts = 2  # different models

range_min = 0    #min value of X
range_max = 10    #max value of X
train_percentage = 0.8 #train-test split




'''
Splitting the dataset considering the numer of workers and rounds
'''
def splitDataset(dataset_X, num_workers, num_rounds):
    
    # total number of sets to generate
    a = num_workers * num_rounds
    
    b = math.floor(len(dataset_X)/a)
    x = len(dataset_X) / b
    
        
    return np.array_split(dataset_X, x)



'''
Two main IF:
- model_drift is TRUE: this means that the method generated two datasets, one for the first [num_rounds/2] rounds, and one for the remaining rounds.
- multi_features is TRUE: return a dataset with three inputs, and one output.
'''
def synthetic_dataset_creator(dataset_size, num_workers, num_rounds, multi_features=False, model_drift=False):

    train_list_X = [] 
    train_list_y = []
    test_X = []
    test_y = []
    
    if model_drift :
        
        after_drift=False
        
        for i in range(drifts):
            tr_X, tr_y, t_X, t_y = generate_data(int(dataset_size/drifts), 
                                                 num_workers, 
                                                 int(num_rounds/drifts), 
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
    
    if multi_features is True:
        
        dataset_X1 = np.random.uniform(low=range_min, high=range_max, size=(dataset_size,))
        dataset_X2 = np.random.uniform(low=range_min, high=range_max, size=(dataset_size,))
        dataset_X3= np.random.uniform(low=range_min, high=range_max, size=(dataset_size,))

        np.random.shuffle(dataset_X1)
        np.random.shuffle(dataset_X2)
        np.random.shuffle(dataset_X3)
        
        dataset_X = np.array([dataset_X1, dataset_X2, dataset_X3])
        
        
        if after_drift:
            if function_type == 'linear':
                dataset_y = dataset_X1 * mm[0] + dataset_X2 * mm[1] + dataset_X3 * mm[2] + mm[3] + np.random.randn(dataset_X1.size) * math.sqrt(v)
            else:
                dataset_y = mm[0] * np.sin(dataset_X1*(mm[1]*math.pi)+ dataset_X2*(mm[2]*math.pi) + dataset_X3*(mm[3]*math.pi)) + mm[4] + np.random.randn(dataset_X1.size) * math.sqrt(v)
                
        else:
            if function_type == 'linear': 
                dataset_y = dataset_X1 * m[0] + dataset_X2 * m[1] + dataset_X3 * m[2] + m[3] + np.random.randn(dataset_X1.size) * math.sqrt(v)
            else:
                dataset_y = m[0] * np.sin(dataset_X1*(m[1]*math.pi)+ dataset_X2*(m[2]*math.pi) + dataset_X3*(m[3]*math.pi)) +m[4]+ np.random.randn(dataset_X1.size) * math.sqrt(v)
            
        dataset_y = dataset_y.reshape(-1,1)
        dataset_X = dataset_X.transpose()
        
    else:
        
        dataset_X = np.random.uniform(low=range_min, high=range_max, size=(dataset_size,))
        np.random.shuffle(dataset_X)
        
        if after_drift:
            if function_type == 'linear':
                dataset_y =  dataset_X * mm[0] + mm[1] +  np.random.randn(dataset_X.size) * math.sqrt(v)
            else:
                dataset_y = mm[0] * np.sin(dataset_X*(mm[1]*math.pi)) + mm[2] + np.random.randn(dataset_X.size) * math.sqrt(v)
        else:
            if function_type == 'linear':
                dataset_y =  dataset_X * m[0] + m[1] +  np.random.randn(dataset_X.size) * math.sqrt(v)
            else:
                dataset_y = m[0] * np.sin(dataset_X*(m[1]*math.pi)) + m[2] + np.random.randn(dataset_X.size) * math.sqrt(v)

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



