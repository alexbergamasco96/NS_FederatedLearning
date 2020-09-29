#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:38:44 2020

@author: alex
"""


import numpy as np
import math


def splitDataset(dataset_X, num_workers, num_rounds):
        
    a = num_workers * num_rounds
    
    b = math.floor(len(dataset_X)/a)
    
    x = len(dataset_X) / b
    
        
    return np.array_split(dataset_X, x)
    

    