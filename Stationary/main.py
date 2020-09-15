#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:42:58 2020

@author: alexb
"""

import os
import numpy as np
import pandas as pd

from sklearn import linear_model
import scipy


class Server:
    """
    Central Server in which we compute the aggregated modelm
    """
    
    def __init__(self, 
                 mean,
                 cov,
                 workers, 
                 num_features, 
                 model = linear_model.LinearRegression()
                 ):
        
        self.mean = mean
        self.cov = cov
        self.model = model
        self.workers = workers
        self.active_workers = len(workers)
        self.num_features = num_features
    
    
    def aggregation(self):
        """
        Perform the aggregation with all the coefficients from the workers.
        Average between parameters
        """
        
        sum_mean = np.zeros(shape=self.num_features)
        sum_cov =  np.zeros(shape=self.num_features) #change cov dimension
        
        for i in self.workers:
            sum_mean = sum_mean + i.getMean()
            sum_cov = sum_mean + i.getCov()
        
        self.mean = sum_mean / self.active_workers
        self.cov = sum_cov / self.active_workers
        
        self.model.coef_ = self.mean
        
        
    
    def return_to_workers(self):    
        """
        Return the aggregated parameters to all the workers
        """
        for i in self.workers:
            i.mean = self.mean
            i.cov = self.cov
            i.num_workers = self.active_workers
        
    def evaluate(self, X):
        """
        Server-side evaluation of the model
        """
        return self.model.predict(X)
        

class Worker:
    
    """
    Client istance: it contains the local model
    """
    
    def __init__(self,
                 mean,
                 cov,
                 model=linear_model.LinearRegression(),
                 num_workers = 0
                 ):
        
        self.mean = mean
        self.cov = cov
        self.model = model
        self.num_workers = num_workers
        
        
    def train(self, X, y):
        """
        Given a training set (X,y), the method trains the model, and then aggregate the previous information about
        the model 
        """
        
        self.model.fit(X,y)
        
        #Attribute of sklearn.linear_model.LinearRegression()
        new_mean = self.model.coef_
        
        self.mean = (self.mean * self.num_workers) / (self.num_workers + 1)  + (new_mean) / (self.num_workers + 1)
        
    
    def evaluate(self, X):
        """
        Worker-side evaluation of the model
        """
        self.model.coef_ = self.mean
        return self.model.predict(X)
        
    
    
    def getMean(self): 
        """
        Return the mean of the current predicted model
        """
        return self.mean
    
    
    def getCov(self):
        """
        Return the computer covariance of the current predicted model
        """
        return self.cov
        


"""
def main():
    print("Hello")
    

if __name__ == "__main__":
    main()

"""