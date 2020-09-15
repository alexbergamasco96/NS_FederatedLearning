#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:13:16 2020

@author: alex
"""

class Worker():
    """
    Client istance: it contains the local model
    """
    
    def __init__(self, mean, cov):
        
        self.mean = mean
        self.cov = cov
        
        
        
    def update(self, mean, cov):
        
        self.mean = mean
        self.cov = cov
    
    
    
    def train(self, x, y):
        
        return self.mean, self.cov