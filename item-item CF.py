#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 23:06:48 2018

@author: lixiang
"""

import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.model_selection import train_test_split

ratings = pd.read_csv('ratings.csv')
#item to item collaborate filtering 
train2 ,test2 = train_test_split(ratings,test_size=0.05)
itemmatrix2 = train2.pivot(index='movieId', columns='userId', values='rating')
itemmatrix2 = itemmatrix2.fillna(0)
normitemmatrix2 = itemmatrix2.apply(lambda x: x-x.mean(), axis=1)
itemdistance2 = pd.DataFrame(spatial.distance.squareform(spatial.distance.pdist(normitemmatrix2)))
movieid = itemmatrix2.index
movieid = list(movieid)
model2 = []

for i in range(len(itemdistance2)): #distance use index not id
    d = np.argsort(itemdistance2.iloc[i].values)#sort的是index
    model2.append(d)

def predicts2(i, movie, num, model2):
    if movie not in movieid:
        return
    if i not in list(itemmatrix2):
        return 
    ind = movieid.index(movie)
    movies = model2[ind]#index
    rates = []
    weight = []
    
    for index in movies: 
        if movieid[index] in itemmatrix2[i]:
            if itemmatrix2[i][movieid[index]]!=0 and ind!=index:
                weight.append(itemdistance2[ind][index])
                rates.append(itemmatrix2[i][movieid[index]])
            elif len(rates) == num:
                break
            else:
                continue
        else: 
            continue
    if len(weight)>0:
        return np.average(rates,weights = weight)
    else:
        return 

print(predicts2(1,1293,10,model2))

observation2 = []
prediction2 = []
for row in test2.iterrows():
    observation2.append(row[1][2])
    prediction2.append(predicts2(int(row[1][0]),int(row[1][1]),50,model2))

sqe2 = 0
num2 = 0
for i in range(len(observation2)):
    if prediction2[i] is not None:
        num2 += 1
        sqe2 += abs(prediction2[i]-observation2[i])
print('mean absolute error is:', sqe2/num2)

