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
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy
from torch.autograd import Variable


from Federated.Server.models import *
from Federated.Server.modelFunctional import *


class Client():
    
    def __init__(self, initialLR, model_type, optimizer='SGD'):
        self.current_round = 0 
        self.model_type = model_type # Model Architecture
        self.initialLR = initialLR 
        self.optimizer_type = optimizer # SGD or Adam
        
        self.model, self.criterion, self.optimizer = createModel(input_size=1,
                                                                 output_size=1, 
                                                                 initialLR=initialLR,
                                                                 hidden=64,
                                                                 model_type=self.model_type,
                                                                 optimizer=optimizer)
        self.loss = 0
    
    
    
    def trainMNIST(self, current_round, local_epochs=5):
        '''
        Train method for MNIST dataset
        '''
        self.model.train()
        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = Variable(data), Variable(target)
                self.optimizer.zero_grad()
                output = self.model(data)
                #loss = F.nll_loss(output, target)
                loss = self.criterion(output, target)
                self.loss = loss.item()
                loss.backward()
                self.optimizer.step()
        
        self.current_round += 1
        
        
    def trainCIFAR(self, current_round, local_epochs=5):
        '''
        Train method for CIFAR dataset
        '''
        self.model.train()
        for epoch in range(local_epochs):
        
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = Variable(data), Variable(target)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        
        self.current_round += 1
        
        
     
    def decayLR(self, coeff=1):
        '''
        On-client learning rate decay 
        '''
        with torch.no_grad():
            for g in self.optimizer.param_groups:
                if self.optimizer_type == 'SGD':
                    # Time based decay
                    g['lr'] = ((self.initialLR)*(coeff)) / (1+self.current_round)
                else:
                    g['lr'] = self.initialLR / (1 + math.sqrt(self.current_round))
           
        
    def train(self, current_round, local_epochs=200, batch_size=8):
        
        '''
        Train method for Synthetic Data
        '''
        
        self.model.train()
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
    
    
    def setDataset(self, train_list_X, train_list_y):
        '''
        Define the training set of this client (different from the others
        '''
        self.train_list_X = train_list_X
        self.train_list_y = train_list_y
    
    
    def setParameters(self, newParameters):
        '''
        Setter for Neural Network Parameters
        Useful for common initialization in the first round as suggested by McMahan 2017
        '''
        with torch.no_grad():
            param_index = 0
            for p in self.model.parameters():
                p.data = newParameters[param_index].data.detach().clone()
                param_index += 1
    
    
    def setLR(self, coeff):
        '''
        Learning rate setter
        '''
        with torch.no_grad():
            for g in self.optimizer.param_groups:
                g['lr'] = coeff
    
    
    
    def getModel(self):
        return self.model
    
    
    
    def setModel(self, model):
        self.model = model
    
    def getParameters(self):
        return list(self.model.parameters())
    