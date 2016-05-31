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
dealData = pd.read_csv("COOLEY-VC_TOTAL_DEAL_VOLUME.csv")
indexData = pd.read_csv("NASDAQOMX-NDXT.csv")
exitData = pd.read_csv("NVCA-VC_EXITS_QUARTERLY.csv")

#get the dates
dealDates = dealData.iloc[:,0]
indexDates = indexData.iloc[:,0]
exitDates = exitData.iloc[:,0]

#get dates for both deal and exit data
datesBoth = np.zeros(len(dealDates)).astype(str)
for i in range(len(dealDates)):
    for j in range(len(exitDates)):
        if (dealDates.iloc[i] == exitDates.iloc[j]):
            datesBoth[i] = dealDates.iloc[i]
datesBoth = datesBoth[datesBoth != '0.0']

#get index dates
dates3 = np.zeros(len(datesBoth)).astype(str)
for i in range(len(datesBoth)):
    for j in range(len(indexDates)):
        if(datesBoth[i] == indexDates.iloc[j]):
            dates3[i] = datesBoth[i]
dates3 = dates3[dates3 != '0.0']

#get the index values for the deal data
dealIndexes = np.zeros(len(dealDates))
for i in range(len(dealDates)):
    for j in range(len(dates3)):
        if(dealDates.iloc[i] == dates3[j]):
            dealIndexes[i] = i
dealIndexes = dealIndexes[dealIndexes > 0]

#get the index values for exit data
exitIndexes = np.zeros(len(exitDates))
for i in range(len(exitDates)):
    for j in range(len(dates3)):
        if(exitDates.iloc[i] == dates3[j]):
            exitIndexes[i] = i

#have to do this since index 0 is needed for exit data
exitIndexes[0] = 1
exitIndexes = exitIndexes[exitIndexes > 0]
exitIndexes[0] = 0.

#get the average MA values
averageMAList = np.zeros(len(dates3))
for i in range(len(exitIndexes)):
    index = int(exitIndexes[i])
    averageMAList[i] = exitData.iloc[index, 4]
    
#get the average IPO values
averageIPOList = np.zeros(len(dates3))
for i in range(len(exitIndexes)):
    index = int(exitIndexes[i])
    averageIPOList[i] = exitData.iloc[index, 7]
    
#get the values of the index    
indexVals = np.zeros(len(dates3))
for i in range(len(exitIndexes)):
    index = int(exitIndexes[i])
    indexVals[i] = indexData.iloc[index, 1]

#get the amount raised    
raisedVals = np.zeros(len(dates3))
for i in range(len(exitIndexes)):
    index = int(exitIndexes[i])
    raisedVals[i] = dealData.iloc[index, 2]

#reshape the data    
raisedVals.shape = 25,1
indexVals.shape = 25,1
averageIPOList.shape = 25,1
averageMAList.shape = 25,1
singleColumn = np.ones(25)
singleColumn.shape = 25,1

#convert the final data to a matrix
finalMat = np.matrix(np.hstack([singleColumn, raisedVals,averageIPOList, 
                                averageMAList, indexVals]))

#convert to a Data Frame
finalDF = pd.DataFrame(finalMat)
finalDF.columns = ["Single_Column", "Amount_Raised", "Average_IPO_Value", 
"Average_MA_Value", "Index_Value"]

testArray = np.array(finalMat)



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

#Linear Regression using Sci-Kit Learn
regr = linear_model.LinearRegression()
regr.fit(X_normTrain, y_train)

