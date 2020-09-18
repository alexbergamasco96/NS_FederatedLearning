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
                 workers, 
                 num_features, 
                 model = linear_model.LinearRegression(),
                 round_ = 1
                 ):
        
        self.mean = mean
        self.model = model
        self.workers = workers
        self.active_workers = len(workers)
        self.num_features = num_features
        self.current_round = round_
        self.aggregation_parameter = self.active_workers
    
    
    def aggregation(self):
        """
        Perform the aggregation with all the coefficients from the workers.
        Average between parameters
        """
        
        sum_mean = np.zeros(shape=self.num_features)
        
        for i in self.workers:
            print(i.getMean())
            sum_mean = sum_mean + i.getMean()
        
        
        if self.current_round == 1:
            
            # First round, no past-information. Aggregate only parameters coming from workers
            self.mean = sum_mean / self.active_workers
            self.current_round += 1
            
        else:
            # Starting from the second round, we consider also past information
            
            self.mean = (self.mean * self.aggregation_parameter + sum_mean) / (self.aggregation_parameter + self.active_workers)
            self.aggregation_parameter += self.active_workers
            self.current_round += 1
        
        self.model.coef_ = self.mean
        
        
    
    def return_to_workers(self):    
        """
        Return the aggregated parameters to all the workers
        """
        for i in self.workers:
            i.mean = self.mean

        
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
                 model=linear_model.LinearRegression()
                 ):
        
        self.mean = mean
        self.model = model
        self.current_mean = mean
        
        
    def train(self, X, y):
        """
        Given a training set (X,y), the method trains the model, and then aggregate the previous information about
        the model 
        """
        
        self.model.fit(X,y)
        
        #Sklearn.linear_model.LinearRegression() attribute
        self.current_mean =  self.model.coef_
        return self.current_mean
    
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
        return self.current_mean
    
        

"""
def main():
    print("Hello")
    

if __name__ == "__main__":
    main()

"""