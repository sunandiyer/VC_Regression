# -*- coding: utf-8 -*-
"""
Created on Sun May 29 18:20:55 2016

@author: sunandiyer
File contains linear regression functions
"""

import numpy as np

#create gradient descent function
def gradientDescent(X,y,theta,alpha,num_iters):
    
    m = len(y)
    costArray = np.zeros(num_iters)

    for i in range(num_iters):
        #calculate hypothesis
        h = np.dot(X,theta)
        #calculate cost
        J = (1/(2*m))*sum(np.dot((np.dot(X,theta) - y).transpose(), np.dot(X,theta) - y))
        costArray[i] = J
        #calculate loss
        loss = h - y
        gradient = np.dot(X.transpose(),loss)/m
        theta = theta - alpha*gradient
    
    return (theta, costArray)


#create feature normalization function
def featureNorm(X):
    
    X_norm = X
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    for i in range(1,X.shape[1]):
        mu[i] = np.mean(X_norm[:,i])
        sigma[i] = np.std(X_norm[:,i])
        X_norm[:,i] = (X_norm[:,i] - mu[i])/sigma[i]
    return X_norm, mu, sigma

def normNewVals(X_test, mu, sigma):
    
    for i in range(len(X_test)):
        for j in range(1,X_test.shape[1]):
            X_test[i,j] = (X_test[i,j] - mu[j])/sigma[j]
    
    return X_test
    
    