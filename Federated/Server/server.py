#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:06:57 2020

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

from Federated.Client.client import *
from Federated.Utils.datasetUtils import *
from Federated.Server.models import *
from Federated.Server.modelFunctional import *

from Federated.Utils import settings

class Server():
    
    def __init__(self, num_workers, model_type, aggregation_method):
        
        self.num_workers = num_workers
        self.model_type = model_type
        self.aggregation_method = aggregation_method
        self.model, self.criterion, self.optimizer = createModel(input_size=1,
                                                                 output_size=1, 
                                                                 initialLR=settings.initialLR,
                                                                 hidden=settings.hiddenNeurons,
                                                                 model_type=self.model_type)
        self.clients = self.generateWorkers(self.num_workers, self.model_type, initialLR=settings.initialLR)
        self.current_round = 0
        self.LRdecay = settings.LRdecay
        
        
        
        
        
    def generateWorkers(self, num_workers, model_type, initialLR):
        
        client_list = []
        for i in range(num_workers):
            client_list.append(Client(initialLR, model_type=self.model_type))
        
        return client_list
    
    
    
    def generateDataset(self, dataset_size, num_rounds, multi_features=False, model_drift=False):
        
        self.initializeComputation(num_rounds)
        self.datasetGenerator = DatasetGenerator(num_workers=self.num_workers, num_rounds=num_rounds)
        train_list_X, train_list_y, self.test_X, self.test_y = self.datasetGenerator.generate(dataset_size, multi_features, model_drift)
        
        for i in range(self.num_workers):
            x = [] 
            y = []
            
            for k in range(self.num_rounds):
                x.append(train_list_X[k*self.num_workers + i])
                y.append(train_list_y[k*self.num_workers + i])
            
            self.clients[i].setDataset(x, y)
        
        self.current_round = 0
        self.c = 0
    
    def initializeComputation(self, num_rounds):
        self.num_rounds = num_rounds
        self.current_round= 0
   
    
    
    def getClientsParameters(self):
        params = []
        for i in self.clients:
            params.append(i.getParameters())
            
        return params
        
    
    def sendParams(self):
        for i in self.clients:
            i.setParameters(list(self.model.parameters()))
    
    def setCurrentParameters(self, newParameters):
        with torch.no_grad():
            param_index = 0
            for p in self.model.parameters():
                p.data = newParameters[param_index].data.detach().clone()
                param_index += 1
    
    
    
    def aggregate(self):
        
        if self.aggregation_method == "FedREG": #FedREG
            newParameters = self.aggregateWithHistory()
        elif self.aggregation_method == 'FedREG_Distance': #Adaptive FedREG
            newParameters = self.aggregateWithDistance()
        else: #FedAVG
            newParameters = self.averageParameters()
            self.lrdecay()
        
        self.setCurrentParameters(newParameters)
        
        
    def averageParameters(self):
        '''
        FedAVG
        '''
        params = self.getClientsParameters()
        newParameters = calculateFedAVGParams(self.num_workers, params)    
        
        return newParameters
        
    
    def aggregateWithHistory(self):
        '''
        FedREG
        '''
        params = self.getClientsParameters()
        params_sum = paramsSum(self.num_workers, params)
        newParameters = calculateFedREGParams(self.num_workers, self.getCurrentParameters() , params_sum, self.c)
        self.c += self.num_workers
        return newParameters
    
    
    def aggregateWithDistance(self):
        '''
        Adaptive FedREG
        '''
        params = self.getClientsParameters()
        params_sum = paramsSum(self.num_workers, params)
        newParameters, beta = calculateFedREGParamsWithAdaption(self.num_workers, self.getCurrentParameters(), params_sum, self.current_round, self.c)
        self.c = self.c * beta + self.num_workers
        return newParameters
    
    
    def fullTraining(self):
        error_list = []
        score_list = []
        for i in range(self.current_round, self.datasetGenerator.num_rounds):
            self.train()
            self.aggregate()
            error, score = self.test()
            error_list.append(error)
            score_list.append(score)
        return error_list, score_list    
    
    
    def train(self):
        self.sendParams()
        for i in self.clients:
            i.train(current_round=self.current_round, local_epochs=200)
            
        self.current_round += 1
        
    
    def test(self):
        
        if self.datasetGenerator.model_drift:
            if self.current_round < (self.datasetGenerator.num_rounds/2):
                current_test_X = self.test_X[0]
                current_test_y = self.test_y[0]
            else:
                current_test_X = self.test_X[1]
                current_test_y = self.test_y[1]
        else:
            current_test_X = self.test_X
            current_test_y = self.test_y
        
        with torch.no_grad():
            predicted = self.model(torch.from_numpy(current_test_X).float()).data.numpy()
            error = (mean_squared_error(current_test_y, predicted))
            score = (r2_score(current_test_y, predicted))
        
        return error, score
    
    
    def lrdecay(self):
        if self.LRdecay:
            for i in self.clients:
                i.decayLR()
            
    
    def getCurrentParameters(self):
        return list(self.model.parameters())

                
    
        
        
        