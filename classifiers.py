#!/usr/bin/python
# -*- coding: utf-8 -*-
from statistics import mode, StatisticsError
from operator import itemgetter
import warnings

import numpy as np
import model

def random (vect, models):
    distances = []
    for ii in range(len(models)):
        for jj in range(models[ii].size()):
            dist = np.linalg.norm(models[ii](jj) - vect)   # Euclidean distance. Manhattan distance is also a good choice
            distances.append([dist,ii])

    # Do nothing with the distances, return a random model
    return np.random.randint(0,len(models))


##########################################
# TODO: Add function to classify 
##########################################
def knn (vect, models, k=1, ord=2):
    distances = []
    for ii in range(len(models)):
        for jj in range(models[ii].size()):
            dist = np.linalg.norm(models[ii](jj) - vect, ord=ord)   # Euclidean distance. Manhattan distance is also a good choice
            distances.append([dist,ii])
    distances.sort()
    
    if k==1:
        return distances[0][1]
    else:
        m = np.asarray(distances)
        best_m = m[0:k,1] # k nearest models
        try:
            print(m[0:k,:])
            hypotesis = mode(best_m.tolist()); #mode i.e most common value
        except StatisticsError: #there is a tie
            hypotesis = best_m[0] #nearest
            warnings.warn('Empate.')
        return int(hypotesis)
