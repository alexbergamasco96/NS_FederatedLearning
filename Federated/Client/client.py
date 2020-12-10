#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:10:28 2020

@author: alex
"""

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


from Federated.Server.models import *
from Federated.Server.modelFunctional import *


class Client():
    
    def __init__(self, initialLR, model_type):
        self.current_round = 0
        self.model_type = model_type
        self.initialLR = initialLR
        self.model, self.criterion, self.optimizer = createModel(input_size=1,
                                                                 output_size=1, 
                                                                 initialLR=initialLR,
                                                                 hidden=64,
                                                                 model_type=self.model_type)
        
    
    
    
    def setDataset(self, train_list_X, train_list_y):
        self.train_list_X = train_list_X
        self.train_list_y = train_list_y
    
    
    
    def getModel(self):
        return self.model
    
    
    
    def setModel(self, model):
        self.model = model
    
    def getParameters(self):
        return list(self.model.parameters())
    
    def setParameters(self, newParameters):
        with torch.no_grad():
            param_index = 0
            for p in self.model.parameters():
                p.data = newParameters[param_index].data.detach().clone()
                param_index += 1
        
    # SISTEMA
    def train(self, current_round, local_epochs=200, batch_size=8):
        
        for epoch in range(local_epochs):
        
            permutation = torch.randperm(self.train_list_X[current_round].size()[0])
            
                 
            for i in range(0, self.train_list_X[current_round].size()[0], batch_size):
                
                self.optimizer.zero_grad()
    
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = self.train_list_X[current_round][indices], self.train_list_y[current_round][indices]
                
                y_pred = self.model.forward(batch_x.float())
                loss = self.criterion(y_pred,batch_y.float())
    
                loss.backward()
                self.optimizer.step()
        
        
        
        self.current_round += 1
        
        
    def decayLR(self):
        with torch.no_grad():
            for g in self.optimizer.param_groups:
                g['lr'] = self.initialLR / (1+self.current_round)
    
    