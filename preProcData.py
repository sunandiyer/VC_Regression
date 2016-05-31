# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:58:13 2016

@author: sunandiyer
This file is used to preprocess the data
"""

import numpy as np
import pandas as pd

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

finalDF.to_csv("output.csv", sep = ',', index = False)

