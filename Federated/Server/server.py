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
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy
from torch.autograd import Variable

import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset   
torch.backends.cudnn.benchmark=True

from torch.autograd import Variable



from Federated.Client.client import *
from Federated.Utils.datasetUtils import *
from Federated.Server.models import *
from Federated.Server.modelFunctional import *

from Federated.Utils import settings

class Server():
    
    def __init__(self, num_workers, model_type, aggregation_method, optimizer='SGD', LRdecay=True):
        
        self.num_workers = num_workers
        self.model_type = model_type
        self.aggregation_method = aggregation_method
        self.model, self.criterion, self.optimizer = createModel(input_size=1,
                                                                 output_size=1, 
                                                                 initialLR=settings.initialLR,
                                                                 hidden=settings.hiddenNeurons,
                                                                 model_type=self.model_type,
                                                                 optimizer=optimizer)
        self.clients = self.generateWorkers(self.num_workers, self.model_type, optimizer=optimizer, initialLR=settings.initialLR)
        self.current_round = 0
        self.LRdecay = settings.LRdecay
        self.mean = self.initializeMean() # Adaptive-FedAVG mean initialization
        self.previous_mean = copy.deepcopy(self.mean)
        self.previous_mean_loss = 0 
        self.mean_loss = 0 # Adaptive-FedAVG loss-based mean initialization
        self.variance = 0 # Adaptive-FedAVG variance initialization
        self.previous_variance = 0
        self.LRcoeff = 0
        self.c = 0 # FedREG coefficient
        
        
    
    def initializeMean(self):
        '''
            Initialize the mean vector of parameters to 0
        '''
        with torch.no_grad():
            a = np.array([])
            for i in self.model.parameters():
                for j in i:
                    a = np.append(a, j.clone().cpu())
            a = np.zeros(len(a))
        
        return a
                
        
        
    def generateWorkers(self, num_workers, model_type, optimizer, initialLR):
        '''
        Generate the set of workers, with same model and optimizer
        '''
        client_list = []
        for i in range(num_workers):
            client_list.append(Client(initialLR, optimizer=optimizer, model_type=self.model_type))
        
        return client_list
    
    
    def generateSyntheticDataset(self, dataset_size, num_rounds, multi_features=False, model_drift=False):
        '''
        Synthetic dataset generator, for synthetic tests
        '''
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
    
    
    
    def generateMNISTDataset(self, num_rounds):
        '''
        Setting Distributed MNIST dataset for Class-Introduction concept drift
        '''
        self.num_rounds = num_rounds
        
        # Image Modification 
        transform_train = transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))]
                                       )
        
        # Loading MNIST using torchvision.datasets
        traindata = torchvision.datasets.MNIST('./data', train=True, download=True,
                               transform= transform_train)
        
        
        if self.current_round < (self.num_rounds/2) :
            remove_list = settings.remove_list
        else:
            remove_list = []
            
        traindata = trainFiltering(traindata, remove_list, self.num_workers)
        
        # Dividing the training data into num_clients, with each client having equal number of images
        traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / self.num_workers) for _ in range(self.num_workers)])
        
        # Creating a pytorch loader for a Deep Learning model
        train_loader = [torch.utils.data.DataLoader(x, batch_size=settings.batch_size, shuffle=True) for x in traindata_split]
        
        
        
        testdata = torchvision.datasets.MNIST('./data', train=False, transform=transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))]
                                       ))
        
        
        len_test_dataset = int(len(traindata) / 3)
        testdata = testFiltering(testdata, remove_list, len_test_dataset)
        
        # Loading the test iamges and thus converting them into a test_loader
        self.test_loader = torch.utils.data.DataLoader( testdata, batch_size=settings.batch_size, shuffle=True)
        
        for i in range(len(train_loader)):
            self.clients[i].train_loader = train_loader[i]
            
            
    def generateMNISTDatasetTwoClasses(self, num_rounds):
        '''
        Setting Distributed MNIST dataset for Two-Class swap concept drift
        '''
        self.num_rounds = num_rounds
        
        # Image modification 
        transform_train = transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))]
                                       )
        
        # Loading MNIST using torchvision.datasets
        traindata = torchvision.datasets.MNIST('./data', train=True, download=True,
                               transform= transform_train)
        
        
        remove_list = [2,3,4,5,6,7,8,9]
            
        traindata = trainFiltering(traindata, remove_list, self.num_workers)
        
        if self.current_round >= (self.num_rounds/2) :
            traindata.targets.apply_(lambda x: 1 if x==0 else (0 if x==1 else x))
        
        # Dividing the training data into num_clients, with each client having equal number of images
        traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / self.num_workers) for _ in range(self.num_workers)])
        
        # Creating a pytorch loader for a Deep Learning model
        train_loader = [torch.utils.data.DataLoader(x, batch_size=settings.batch_size, shuffle=True) for x in traindata_split]
        
        
        
        testdata = torchvision.datasets.MNIST('./data', train=False, transform=transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))]
                                       ))
        
        
        len_test_dataset = int(len(traindata) / 3)
        testdata = testFiltering(testdata, remove_list, len_test_dataset)
        
        if self.current_round >= (self.num_rounds/2) :
            testdata.targets.apply_(lambda x: 1 if x==0 else 0)
        
        # Loading the test iamges and thus converting them into a test_loader
        self.test_loader = torch.utils.data.DataLoader( testdata, batch_size=settings.batch_size, shuffle=True)
        
        for i in range(len(train_loader)):
            self.clients[i].train_loader = train_loader[i]
            
            
            
    def generateMNISTDatasetSwapClasses(self, num_rounds):
        '''
        Setting Distributed MNIST dataset for Class-Swap concept drift
        '''
        self.num_rounds = num_rounds
        
        # Image modification 
        transform_train = transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))]
                                       )
        
        # Loading MNIST using torchvision.datasets
        traindata = torchvision.datasets.MNIST('./data', train=True, download=True,
                               transform= transform_train)
        
        
        remove_list = []
            
        traindata = trainFiltering(traindata, remove_list, self.num_workers)
        
        if self.current_round >= (self.num_rounds/2) :
            traindata.targets.apply_(lambda x: 1 if x==0 else (0 if x==1 else x))
        
        # Dividing the training data into num_clients, with each client having equal number of images
        traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / self.num_workers) for _ in range(self.num_workers)])
        
        # Creating a pytorch loader for a Deep Learning model
        train_loader = [torch.utils.data.DataLoader(x, batch_size=settings.batch_size, shuffle=True) for x in traindata_split]
        
        
        
        testdata = torchvision.datasets.MNIST('./data', train=False, transform=transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))]
                                       ))
        
        
        len_test_dataset = int(len(traindata) / 3)
        testdata = testFiltering(testdata, remove_list, len_test_dataset)
        
        if self.current_round >= (self.num_rounds/2) :
            testdata.targets.apply_(lambda x: 1 if x==0 else (0 if x==1 else x))
        
        # Loading the test iamges and thus converting them into a test_loader
        self.test_loader = torch.utils.data.DataLoader( testdata, batch_size=settings.batch_size, shuffle=True)
        
        for i in range(len(train_loader)):
            self.clients[i].train_loader = train_loader[i]
    
    
    
    
    def generateCIFARDataset(self, num_rounds):
        
        '''
        Setting Distributed CIFAR dataset for Class-Introduction concept drift
        '''
        
        self.num_rounds = num_rounds
        
        # Image transformation 
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Loading CIFAR10 using torchvision.datasets
        traindata = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                               transform= transform_train)
        
        
        if self.current_round < (self.num_rounds/2) :
            remove_list = settings.remove_list
        else:
            remove_list = []
            
        
        traindata.targets = torch.LongTensor(traindata.targets)
        traindata = trainFiltering(traindata, remove_list, self.num_workers)
        
        # Dividing the training data into num_clients, with each client having equal number of images
        traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / self.num_workers) for _ in range(self.num_workers)])
        
        # Creating a pytorch loader for a Deep Learning model
        train_loader = [torch.utils.data.DataLoader(x, batch_size=settings.batch_size, shuffle=True) for x in traindata_split]
        
        
        # Importing test data
        testdata = torchvision.datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]))
        
        # Changing type of data.target (in CIFAR10 is a list, not a tensor)
        testdata.targets = torch.LongTensor(testdata.targets)
            
        len_test_dataset = int(len(traindata) / 3)
        testdata = testFiltering(testdata, remove_list, len_test_dataset)
        
        # Converting images into a test_loader
        self.test_loader = torch.utils.data.DataLoader( testdata, batch_size=settings.batch_size, shuffle=True)
        
        # setting the train_loaders client-side
        for i in range(len(train_loader)):
            self.clients[i].train_loader = train_loader[i]
            
            
            
    def generateCIFARDatasetTwoClasses(self, num_rounds):
        
        '''
        Setting Distributed CIFAR dataset for Two-Class swap concept drift
        '''
        
        self.num_rounds = num_rounds
        
        # Image transformation
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Loading CIFAR10 using torchvision.datasets
        traindata = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                               transform= transform_train)
        
        
        traindata.targets = torch.LongTensor(traindata.targets) 
        
        remove_list = [2,3,4,5,6,7,8,9]
        
        if self.current_round >= (self.num_rounds/2) :
            traindata.targets.apply_(lambda x: 1 if x==0 else (0 if x==1 else x))
        
        traindata = trainFiltering(traindata, remove_list, self.num_workers)
        
        
        
        # Dividing the training data into num_clients, with each client having equal number of images
        traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / self.num_workers) for _ in range(self.num_workers)])
        
        # Creating a pytorch loader for a Deep Learning model
        train_loader = [torch.utils.data.DataLoader(x, batch_size=settings.batch_size, shuffle=True) for x in traindata_split]
        
        
        # Importing test data
        testdata = torchvision.datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]))
        
        # Changing type of data.target (in CIFAR10 is a list, not a tensor)
        testdata.targets = torch.LongTensor(testdata.targets)
        
        len_test_dataset = int(len(traindata) / 3)
        testdata = testFiltering(testdata, remove_list, len_test_dataset)
        
        if self.current_round >= (self.num_rounds/2) :
            testdata.targets.apply_(lambda x: 1 if x==0 else (0 if x==1 else x))
        
        # Converting images into a test_loader
        self.test_loader = torch.utils.data.DataLoader( testdata, batch_size=settings.batch_size, shuffle=True)
        
        # setting the train_loaders client-side
        for i in range(len(train_loader)):
            self.clients[i].train_loader = train_loader[i]
            
    
    def generateCIFARDatasetSwapClasses(self, num_rounds):
        
        '''
        Setting Distributed CIFAR dataset for Class-Swap concept drift
        '''
        
        self.num_rounds = num_rounds
        
        # Image transformation (No Augmentation)
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Loading CIFAR10 using torchvision.datasets
        traindata = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                               transform= transform_train)
        
        traindata.targets = torch.LongTensor(traindata.targets)
        
        if self.current_round >= (self.num_rounds/2) :
            traindata.targets.apply_(lambda x: 1 if x==0 else (0 if x==1 else x))
        
        remove_list = []
            
        
        traindata = trainFiltering(traindata, remove_list, self.num_workers)
        
        # Dividing the training data into num_clients, with each client having equal number of images
        traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / self.num_workers) for _ in range(self.num_workers)])
        
        # Creating a pytorch loader for a Deep Learning model
        train_loader = [torch.utils.data.DataLoader(x, batch_size=settings.batch_size, shuffle=True) for x in traindata_split]
        
        
        # Importing test data
        testdata = torchvision.datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]))
        
        # Changing type of data.target (in CIFAR10 is a list, not a tensor)
        testdata.targets = torch.LongTensor(testdata.targets)
            
        if self.current_round >= (self.num_rounds/2) :
            testdata.targets.apply_(lambda x: 1 if x==0 else (0 if x==1 else x))
        
        len_test_dataset = int(len(traindata) / 3)
        testdata = testFiltering(testdata, remove_list, len_test_dataset)
        
        # Converting images into a test_loader
        self.test_loader = torch.utils.data.DataLoader( testdata, batch_size=settings.batch_size, shuffle=True)
        
        # setting the train_loaders client-side
        for i in range(len(train_loader)):
            self.clients[i].train_loader = train_loader[i]
        
        
    
    def initializeComputation(self, num_rounds):
        self.num_rounds = num_rounds
        self.current_round= 0
        
    
    def aggregate(self):
        
        if self.aggregation_method == "FedREG": #FedREG
            newParameters = self.aggregateWithHistory()
        elif self.aggregation_method == 'FedREG_Distance': #Adaptive FedREG
            newParameters = self.aggregateWithDistance()
        elif self.aggregation_method == 'FedAda':
            newParameters = self.averageParameters()
            self.adaptiveLR(newParameters)
        elif self.aggregation_method == 'FedAda_loss_based':
            newParameters = self.averageParameters()
            loss = self.averageLoss()
            self.adaptiveLR_Loss_Based(loss)
        else: #FedAVG
            newParameters = self.averageParameters()
            # Decay of Learning Rate
            self.lrdecay()
        
        self.setCurrentParameters(newParameters)
        
        
    def averageParameters(self):
        '''
        FedAVG
        '''
        params = self.getClientsParameters()
        newParameters = calculateFedAVGParams(self.num_workers, params)    
        
        return newParameters
    
    
    def averageLoss(self):
        '''
        Retrieve local losses and perform the average
        '''
        losses = self.getClientLosses()
        loss = np.mean(losses)
        
        return loss
    
    
    def adaptiveLR(self, newParameters):
        
        # Create the array containing newParameters
        newParameters_arr = np.array([])
        for i in newParameters:
            for j in i:
                newParameters_arr = np.append(newParameters_arr, j.clone().cpu().data.numpy())
                
        # EMA on the mean
        self.mean = self.previous_mean * settings.beta1 + (1-settings.beta1)*newParameters_arr
        
        # Initialization Bias correction
        self.mean = self.mean / (1-pow(settings.beta1, self.current_round+1))
        
        
        # EMA on the Variance
        self.variance = self.previous_variance * settings.beta2 + (1 - settings.beta2)*np.mean((newParameters_arr-self.previous_mean)*(newParameters_arr-self.previous_mean))
        
        self.previous_mean = copy.deepcopy(self.mean)
        
        temp = copy.deepcopy(self.previous_variance)
        self.previous_variance = copy.deepcopy(self.variance)
        # Initialization Bias correction
        self.variance = self.variance / (1-pow(settings.beta2, self.current_round+1))
        
        if self.current_round < 2:
            r = 1
        else:
            r = np.abs(self.variance / (temp/(1-pow(settings.beta2, self.current_round))))
        
        self.LRcoeff = self.LRcoeff * settings.beta3 + (1-settings.beta3)*r
        
        coeff = self.LRcoeff/ (1-pow(settings.beta3,self.current_round +1))
        
        
        ### SERVER-SIDE LEARNING RATE SCHEDULER 
        
        #No Decay
        #coeff = min(settings.initialLR, settings.initialLR*coeff)
        
        #Decay of 1/t TIME BASED DECAY as in the convergence analysis of FedAVG
        coeff = min(settings.initialLR, (settings.initialLR*coeff)/(self.current_round+1))
        
        #Decay of 0.99
        #coeff = min(settings.initialLR, settings.initialLR*coeff*math.pow(0.99, self.current_round))
        
        
        # Setting the new LR 
        for i in self.clients:
            i.setLR(coeff)
            
            
            
            
    def adaptiveLR_Loss_Based(self, loss):
                
        '''
        Loss-Based Adaptive FedAVG solution
        '''
            
        # EMA on the mean
        self.mean_loss = self.previous_mean_loss * settings.beta1 + (1-settings.beta1)*loss
        
        # Initialization Bias correction
        self.mean_loss = self.mean_loss / (1-pow(settings.beta1, self.current_round+1))
        
        
        # EMA on the Variance
        self.variance = self.previous_variance * settings.beta2 + (1 - settings.beta2)*(loss-self.previous_mean_loss)*(loss-self.previous_mean_loss)
        
        self.previous_mean_loss = copy.deepcopy(self.mean_loss)
        
        temp = copy.deepcopy(self.previous_variance)
        self.previous_variance = copy.deepcopy(self.variance)
        # Initialization Bias correction
        self.variance = self.variance / (1-pow(settings.beta2, self.current_round+1))
        
        if self.current_round < 2:
            r = 1
        else:
            r = np.abs(self.variance / (temp/(1-pow(settings.beta2, self.current_round))))
        
        self.LRcoeff = self.LRcoeff * settings.beta3 + (1-settings.beta3)*r
        
        coeff = self.LRcoeff/ (1-pow(settings.beta3,self.current_round +1))
        
        
        
        ### SERVER SCHEDULER Choosing the decay
        
        #No Decay
        #coeff = min(settings.initialLR, settings.initialLR*coeff)
        
        #Decay of 1/t TIME BASED DECAY as in the convergence analysis of FedAVG
        coeff = min(settings.initialLR, (settings.initialLR*coeff)/(self.current_round+1))
        
        #Decay of 0.99 per round
        #coeff = min(settings.initialLR, settings.initialLR*coeff*math.pow(0.99, self.current_round))
        
        
        for i in self.clients:
            #i.decayLR(coeff)
            i.setLR(coeff)
        
        
    
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
        '''
        Synthetic dataset training phase
        '''
        error_list = []
        score_list = []
        for i in range(self.current_round, self.datasetGenerator.num_rounds):
            self.train()
            self.aggregate()
            error, score = self.test()
            error_list.append(error)
            score_list.append(score)
            self.current_round += 1
        return error_list, score_list    
    
    
    
    
    def fullTrainingMNIST(self):
        
        error_list = []
        score_list = []
        
        for i in range(self.current_round, self.num_rounds):
            
            # Change this row for a different Concept Drift
            self.generateMNISTDataset(settings.num_rounds)
            self.trainMNIST()
            self.aggregate()
            error, score = self.testMNIST()
            error_list.append(error)
            score_list.append(score)
            
            f = open("log.txt", "a")
            f.write("---Round {}\t".format(self.current_round))
            f.write("Score: {:.4f}\tLoss: {:.4f}\n".format(score, error))
            f.close()
            
            
            self.current_round += 1
            
            
        return error_list, score_list
    
    
    
    
    def fullTrainingCIFAR(self):
        
        error_list = []
        score_list = []
        
        for i in range(self.current_round, self.num_rounds):
            
            # Change this row for a different Concept Drift
            self.generateCIFARDataset(settings.num_rounds)
            self.trainCIFAR()
            self.aggregate()
            error, score = self.testCIFAR()
            error_list.append(error)
            score_list.append(score)
            self.current_round += 1
            
            f = open("log.txt", "a")
            f.write("---Round {}\t".format(self.current_round))
            f.write("Score: {:.4f}\tLoss: {:.4f}\n".format(score, error))
            f.close()
            
            
        return error_list, score_list 
    
    
    def trainMNIST(self):
        self.sendParams()
        for i in self.clients:
            i.trainMNIST(current_round=self.current_round, local_epochs=settings.local_epochs)
    
    
    def trainCIFAR(self):
        self.sendParams()
        for i in self.clients:
            i.trainCIFAR(current_round=self.current_round, local_epochs=settings.local_epochs)
    
    
    def train(self):
        self.sendParams()
        for i in self.clients:
            i.train(current_round=self.current_round, local_epochs=settings.local_epochs)
            
   
        
    
    
    
    def test(self):
        '''
        Synthetic data tester
        '''
       
        self.model.eval()
        
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
    
    
    
    def testMNIST(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = Variable(data), Variable(target)
                output = self.model(data)
                #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
    
        test_loss /= len(self.test_loader.dataset)
        acc = correct / len(self.test_loader.dataset)
        
        
        return test_loss, acc
    
    
    
    def testCIFAR(self):
   
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = Variable(data), Variable(target)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
    
        test_loss /= len(self.test_loader.dataset)
        acc = correct / len(self.test_loader.dataset)
        
        
        return test_loss, acc
    
    
    
    def lrdecay(self):
        '''
        Decay client-side (not managed by a server scheduler
        '''
        if self.LRdecay:
            for i in self.clients:
                i.decayLR()
            
    
    def getCurrentParameters(self):
        return list(self.model.parameters())

                
    def getClientsParameters(self):
        params = []
        for i in self.clients:
            params.append(i.getParameters())
            
        return params
    
    
    def getClientLosses(self):
        losses = []
        for i in self.clients:
            losses.append(i.getLoss())
            
        return losses
        
    
    def sendParams(self):
        for i in self.clients:
            i.setParameters(list(self.model.parameters()))
    
    def setCurrentParameters(self, newParameters):
        with torch.no_grad():
            param_index = 0
            for p in self.model.parameters():
                p.data = newParameters[param_index].data.detach().clone()
                param_index += 1
        
        
        