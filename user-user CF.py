#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 23:05:12 2018

@author: lixiang
"""
import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.model_selection import train_test_split

ratings = pd.read_csv('ratings.csv')
#split the sample data into training set and test set. test size is 10%
train ,test = train_test_split(ratings,test_size=0.05)
#train model
usermatrix1 = train.pivot(index='userId', columns='movieId', values='rating')
usermatrix1 = usermatrix1.fillna(0)
normusermatrix1 = usermatrix1.apply(lambda x: x-x.mean(), axis=1)
userdistance1 = pd.DataFrame(spatial.distance.squareform(spatial.distance.pdist(normusermatrix1,'euclidean')))

model1 = []
for i in range(len(userdistance1)):
    d = np.argsort(userdistance1.iloc[i].values)
    model1.append(d)

def predicts(i, movie, num, model1):
    if(movie not in list(usermatrix1)):
        return 
    
    users = model1[i-1]
    rates = []
    weight = []
    
    for id in users: 
        if usermatrix1[movie][id+1]!=0 and id!=(i-1):
            weight.append(userdistance1[i-1][id])
            rates.append(usermatrix1[movie][id+1])
        elif len(rates) == num:
            break
        else:
            continue
    if len(weight)>0:
        return np.average(rates,weights = weight)
    else:
        return 

#make prediction on test data
observation = []
prediction = []
for row in test.iterrows():
    observation.append(row[1][2])
    prediction.append(predicts(int(row[1][0]),int(row[1][1]),50,model1))
    

sqe = 0
num = 0
for i in range(len(observation)):
    if prediction[i] is not None:
        num += 1
        sqe += abs(prediction[i]-observation[i])
print('mean absolute error is:', sqe/num)