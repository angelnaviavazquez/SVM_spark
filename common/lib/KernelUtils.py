#!/usr/bin/python
# Filename: KernelUtils.py

import numpy as np

#Kernel functions

def kernel(primero,segundo,gamma):
    resta=primero-segundo
    resta**=2.0
    return np.exp(-gamma*((resta).sum()))


def kncVector(elementosbase,candidato,gamma):
    knc=np.zeros((len(elementosbase),1))
    for i in range(len(elementosbase)):
        knc[i,0] = kernel(elementosbase[i],candidato,gamma)
    return knc

def kernelMatrix(setDim1,setDim2,gamma):
    K=np.zeros((len(setDim1),len(setDim2)))
    for i in range(len(setDim1)):
        for e in range(len(setDim2)):
            K[i,e] = kernel(setDim1[i],setDim2[e],gamma)
    return K  
