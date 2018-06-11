from statistics import mode, StatisticsError
import warnings

import numpy as np


def knn(vect, models, k=1, ord=2):
    distances = []
    for ii in range(len(models)):
        for jj in range(models[ii].size()):
            dist = np.linalg.norm(models[ii](jj) - vect, ord=ord)
            distances.append([dist, ii])
    distances.sort()

    if k == 1:
        return distances[0][1]
    else:
        m = np.asarray(distances)
        best_m = m[0:k, 1]  # k nearest models
        try:
            hypotesis = mode(best_m.tolist())  # mode i.e most common value
        except StatisticsError:  # there is a tie
            hypotesis = best_m[0]  # nearest
            warnings.warn('Empate.')
        return int(hypotesis)
