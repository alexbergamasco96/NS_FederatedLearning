#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:08:30 2020

@author: alex
"""


# -------------- Federated Parameters

num_workers = 64        # Number of workers
num_rounds = 40       # Total number of rounds

initialLR = 1e-2

local_epochs = 5     # Epochs performed client-side
batch_size = 32         # Batch_size for local training phase

LRdecay = True        # False (= decay not required) in case of Full Gradient Descent


 
#Type of Neural Network. MNISTFFNN, MNISTCNN, CIFARCNN
model_type= 'MNISTFFNN'


### Classes to remove before the drift in case of Class-Introduction concept drift 

remove_list = [7,8,9]


### Adaptive Parameters, Decay rates of EMAs

beta1 = 0.5  # beta coefficient in parameters EMA
beta2 = 0.5  # beta coefficient in variance EMA
beta3 = 0.5  # beta coefficient in coeff EMA








##############################
## Synthetic dataset Tests, Old code
##############################

adaptive_FedREG = False  # Old Algorithm

dataset_size = 24000    # dataset size, Linear Tests only
multi_features = False  # Multi Feature inputs, Linear Test only
model_drift = True     # Drift of the model after [num_rounds/2] rounds, Linear Tests Only


hiddenNeurons = 64

input_size = 3 if multi_features else 1
output_size = 1








'''
    The function is defined as:    
    
    y = m[0]*sin(X*m[1]*phi)+m[2]
'''

''' Params for Periodic Function'''

m = [10.0, 0.1, 1.5] #before drift
mm = [6.0, 0.15, -3.0] #after drift
#mm = [15.0, 0.2, -3.0] #before drift

''' Params for Linear Function '''
'''
m = [3.0, 0.5, 1.5, 1.0] #before drift
mm = [4.0, 1.0, 0.5, 2.5] #after drift
'''


v = 1 # noise

function_type = 'periodic' # 'linear' or 'periodic'



drifts = 1  # different models

range_min = 0    #min value of X
range_max = 20    #max value of X
train_percentage = 0.8 #train-test split



