# -*- coding: utf-8 -*-
"""
Created on Sun May 29 18:13:41 2016

@author: sunandiyer
This is the project stuyding VC regression
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import linRegFunc as rf
from sklearn.cross_validation import train_test_split
from sklearn import linear_model

#load data from files
data = pd.read_csv("output.csv")
testArray = np.array(data)

#get data ready for gradient descent
X = testArray[:,:4]
y = testArray[:,4]

#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42)

X_normTrain, mu, sigma = rf.featureNorm(X_train)                                                    
y_train.shape = len(y_train),1
theta = np.ones(4)
theta.shape = 4,1

#calculate optimum alpha value
errorVals = np.zeros(100)
for i in range(1,101):
    theta = np.ones(4)
    theta.shape = 4,1
    alpha = i*0.01
    theta, cost = rf.gradientDescent(X_normTrain,y_train,theta,alpha,250)
    errorVals[i-1] = (abs(np.dot(X_normTrain,theta) - y_train)/y).sum()/len(y_train)

optimumAlpha = (np.argmin(errorVals) + 1) * 0.01

theta, cost = rf.gradientDescent(X_normTrain,y_train,theta,optimumAlpha,250)

X_normTest = rf.normNewVals(X_test, mu, sigma)

y_test.shape = len(y_test),1
totError = (abs(np.dot(X_normTest, theta) - y_test)/y_test).sum()/len(y_test)


plt.xlabel("Prediction")
plt.ylabel("Actual Value")
plt.title("Prediction vs. Actual")
plt.scatter(np.dot(X_normTest,theta), y_test)
plt.show()
plt.clf()

#Linear Regression using Sci-Kit Learn
regr = linear_model.LinearRegression()
regr.fit(X_normTrain, y_train)
plt.ylabel("My Prediction")
plt.xlabel("Sci-Kit Learn Prediction")
plt.scatter(regr.predict(X_normTest), np.dot(X_normTest, theta))
plt.show()
