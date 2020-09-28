#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:42:58 2020

@author: alexb
"""

import os
import numpy as np

from sklearn import linear_model
import scipy


class Server:
    """
    Central Server in which we compute the aggregated model
    """
    
    def __init__(self, 
                 coef,
                 intercept,
                 workers, 
                 num_features, 
                 model = linear_model.LinearRegression(),
                 round_ = 1
                 ):
        
        self.coef = coef
        self.model = model
        self.intercept = intercept
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
        
        sum_coef = np.zeros(shape=self.num_features)
        
        sum_intercept = np.zeros(shape=1)
        
        for i in self.workers:
            sum_coef = sum_coef + i.getCoef()
            sum_intercept = sum_intercept + i.getIntercept()
        
        
        # First round, no past-information. Aggregate only parameters coming from workers
        if self.current_round == 1:
            
            self.coef = sum_coef / self.active_workers
            self.intercept = sum_intercept / self.active_workers
            self.current_round += 1
        
        # Starting from the second round, we consider also past information
        else:
            
            self.coef = (self.coef * self.aggregation_parameter + sum_coef) / (self.aggregation_parameter + self.active_workers)
            self.intercept = (self.intercept * self.aggregation_parameter + sum_intercept) / (self.aggregation_parameter + self.active_workers)
            self.aggregation_parameter += self.active_workers
            self.current_round += 1
        
        self.model.coef_ = self.coef.copy()
        self.model.intercept_ = self.intercept.copy()
        
        
    
    def return_to_workers(self):    
        """
        Return the aggregated parameters to all the workers
        """
        for i in self.workers:
            i.coef = self.coef
            i.intercept = self.intercept

        
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
                 coef,
                 intercept,
                 model=linear_model.LinearRegression()
                 ):
        
        self.coef = coef
        self.intercept = intercept
        self.model = model
        self.current_coef = coef
        self.current_intercept = intercept
        
        
    def train(self, X, y):
        """
        Given a training set (X,y), the method trains the model, and then aggregate the previous information about
        the model 
        """
        
        self.model.fit(X,y)
        
        #Sklearn.linear_model.LinearRegression() attribute
        self.current_coef =  self.model.coef_.copy()
        self.current_intercept = self.model.intercept_.copy()
        #return self.current_mean
    
    def evaluate(self, X):
        """
        Worker-side evaluation of the model
        """
        self.model.coef_ = self.coef.copy()
        self.model.intercept_ = self.intercept.copy()
        
        return self.model.predict(X)
        
    
    
    def getCoef(self): 
        """
        Return the coefficients of the current predicted model
        """
        return self.current_coef.copy()
    
    def getIntercept(self):
        """
        Return the intercept coefficient of the current predicted model
        """
        return self.current_intercept.copy()
