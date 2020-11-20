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


train_percentage = 0.8

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


'''
Splitting the dataset considering the numer of workers and rounds
'''
def splitDataset(dataset_X, num_workers, num_rounds):
    
    # total number of sets to generate
    a = num_workers * num_rounds
    
    b = math.floor(len(dataset_X)/a)
    x = len(dataset_X) / b
    
        
    return np.array_split(dataset_X, x)
    


def synthetic_dataset_creator(dataset_size, num_workers, num_rounds, multi_features=False):
    
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