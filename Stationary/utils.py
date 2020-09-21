#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:38:44 2020

@author: alex
"""


import numpy as np


def splitDataset(dataset_X, num_workers, num_rounds):
        
    x = num_workers * num_rounds
        
    return np.array_split(dataset_X, x)
    

    