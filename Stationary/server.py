#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:12:50 2020

@author: alex
"""

import os
import numpy as np
import pandas as pd

import sklearn
import scipy

class Server():
    """
    Central Server in which we compute the aggregated modelm
    """
    
    def __init__(self, mean, cov, workers):
        
        self.mean = mean
        self.cov = cov
        self.workers = workers
        