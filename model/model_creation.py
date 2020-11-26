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

from torch.autograd import Variable



class periodicModel(torch.nn.Module):
    
    def __init__(self, inputSize, outputSize, H = 64):
        
        super(periodicModel, self).__init__()
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

    
class linearModel(torch.nn.Module):
    
    def __init__(self, inputSize, outputSize, H = 20):
        
        super(linearModel, self).__init__()
        self.linear = torch.nn.Linear(inputSize, H, bias=True)
        self.linear2 = torch.nn.Linear(H, outputSize)

        
    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(x)
        return x
    

class nonLinearModel(torch.nn.Module):
    
    def __init__(self, inputSize, outputSize, H = 20):
        
        super(nonLinearModel, self).__init__()
        self.linear = torch.nn.Linear(inputSize, H, bias=True)
        self.linear2 = torch.nn.Linear(H, outputSize)

        
    def forward(self, x):
        x = F.relu(self.linear(x))
        x = self.linear2(x)
        return x
    
    
    
def model_creator(input_size, output_size, num_workers, hidden=64, model_type='periodic'):
    
    w = []
    
    if model_type == 'linear':
        w.append(linearModel(input_size, output_size, H=hidden))
    elif model_type == 'non_linear':
        w.append(nonLinearModel(input_size, output_size, H=hidden))
    else:
        w.append(periodicModel(input_size, output_size, H=hidden))
    
  
    for i in range(1, num_workers):
        w.append(copy.deepcopy(w[0]))
    
    w_avg = []
    for i in range(0, num_workers):
        w_avg.append(copy.deepcopy(w[0]))
        
    return w, w_avg