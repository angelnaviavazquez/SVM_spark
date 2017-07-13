#!/usr/bin/python
# Filename: KernelUtils.py

import numpy as np

from scipy.spatial.distance import squareform,cdist

#Kernel functions

def kernel(primero,segundo,sigma):
    gamma=1.0/(sigma**2.0)
    resta=primero-segundo
    resta**=2.0
    return np.exp(-gamma*((resta).sum()))

def kernelMatrix(setDim1,setDim2,sigma):
    sqrtDists = cdist(setDim1,setDim2,'euclidean')
    gamma=1.0/(sigma**2.0)
    return np.exp(-gamma*np.power(sqrtDists,2.0))
    
