# -*- coding: utf-8 -*-
#!/usr/bin/python
# Filename: IRWLSUtils.py

from KernelUtils import *
import time
import math
from scipy.linalg import cho_factor, cho_solve, solve
from sklearn.datasets import load_svmlight_file
from pyspark.mllib.regression import LabeledPoint   
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from ResultsUtils import *
import numpy as np
from pyspark.mllib.clustering import KMeans, KMeansModel

from collections import defaultdict



def IRWLS_NOT_MEMORY(sc, originaldataset,Bases,C,sigma,Niter=100, stop_criteria=1e-3):
    
    #From labeledPoint to tuples label, kernel vector
    nBases=Bases.shape[0]     
    broadcastBases=sc.broadcast(Bases)

    #dataset.cache()
    dataset = originaldataset.map(lambda x: (x.label,np.array(x.features))).cache()
    
    # Basis kernel matrix
    KC=kernelMatrix(Bases,Bases,sigma)+np.diag(0.000001*np.ones(nBases))
    
    # Weights Initialization
    Beta = 0.001*np.random.rand(nBases,1)
    bestBeta = Beta    
   
    NSVs = -99
    costFunction=float('inf')
    stopCriteria=0

    for i in range(Niter):
        
        tInicioIter = time.time()
        
        # Weighted Least Squares Problem
        (K1,K2) = dataset.map(lambda x: _getK1andK2_NOT_MEMORY(x,broadcastBases,Beta,sigma,C,i,1.0)).reduce(lambda a,b:(a[0]+b[0],a[1]+b[1]))  
        newBeta = solve(K1+KC,K2)

        # Evaluating Best Step
        if i>0:           
            WeightsList=list()
            values=np.array([0.015625,0.03125,0.0625,0.125,0.25,0.5,1.0])
            for value in values:
                WeightsList.append(Beta*(1.0-value)+newBeta*(value))
            hinges = dataset.map(lambda x: _getHinges_NOT_MEMORY(x,broadcastBases,WeightsList,sigma)).reduce(lambda a,b:a+b)

            for indhin in range(len(hinges)):
                hinges[indhin]=hinges[indhin]+0.5*((WeightsList[indhin].transpose().dot(KC.dot(WeightsList[indhin])))[0,0])

            bestIndex=np.argmin(hinges)
            if hinges[bestIndex]<costFunction:
                newBeta=WeightsList[bestIndex]
                costFunction=hinges[bestIndex]
            else:
                newBeta=Beta
                stopCriteria=1
        else:
            WeightsList=list()
            WeightsList.append(bestBeta)
            hinges = dataset.map(lambda x: _getHinges_NOT_MEMORY(x,broadcastBases,WeightsList,sigma)).reduce(lambda a,b:a+b)        
            costFunction=hinges[0]+0.5*((newBeta.transpose().dot(KC.dot(newBeta)))[0,0])


        Beta=newBeta
        bestBeta=Beta
        
        tFinIter = time.time()
        
        print "Iteration",(i+1),": Cost Function",costFunction,", Iteration Time", (tFinIter-tInicioIter)    

        if stopCriteria==1:
            break        
    

    return bestBeta


def _getHinges_NOT_MEMORY(data,broadcastBases,WeighList,sigma):
    hinges=np.zeros(len(WeighList))
    vector = kernelMatrix(np.reshape(data[1],(1,-1)),broadcastBases.value,sigma)
    for index in range(len(WeighList)):
        hinges[index]=math.pow(max(0.0,data[0]*(data[0]-vector.dot(WeighList[index])[0,0])),2.0)
    return hinges


def _isSV(trainingSet,broadcastBases,Beta, C, sigma):
    
    vector = kernelMatrix(np.reshape(np.array(trainingSet.features),(1,-1)),broadcastBases.value,sigma)
    error = trainingSet.label-vector.dot(Beta)[0,0]
    a=C/(error*trainingSet.label) 
    
    if a>0.0:
        return 1
    else:
        return 0


def _getK1andK2_NOT_MEMORY(trainingSet,broadcastBases,Beta,sigma,C,iteration,samplingRate):

    resultado = (0.0,0.0)

    vector = kernelMatrix(np.reshape(trainingSet[1],(1,-1)),broadcastBases.value,sigma)

    if np.random.random()<samplingRate:
        if iteration == 0:
            a=1
        else:
            error = trainingSet[0]-vector.dot(Beta)[0,0]
            a=C/(samplingRate*error*trainingSet[0])   
            if a>100000.0:
                a=100000.0
        if a>0.0:
            resultado = (a*vector.transpose().dot(vector),a*vector.transpose().dot(trainingSet[0]))

    return resultado


def train_SVM(sc, XtrRDD, XtstRDD, sigma, C, NC, stop_criteria=1e-3):

    print "Obtaining weights using kmeans"
    time_ini = time.time()
    Bases = list()
    clusters = KMeans.train(XtrRDD.map(lambda x: x.features), NC, maxIterations=80, initializationMode="random")
    base = np.array(clusters.centers)
    Bases = np.array([np.array(x) for x in base])
    basis_time = time.time() - time_ini
    print "Time obtaining centroids", basis_time

    print "Obtaining weights using IRWLS"
    time_ini = time.time()
    Pesos = IRWLS_NOT_MEMORY(sc, XtrRDD, Bases, C, sigma, stop_criteria=stop_criteria)
    pesos_time = time.time() - time_ini
    print "Time obtaining weights", pesos_time

    broadcastBases = sc.broadcast(Bases)
    NSVs = XtrRDD.map(lambda x: _isSV(x, broadcastBases, Pesos, C, sigma)).reduce(lambda a, b: a + b)

    AUCTR, AUCTST, ACCTR, ACCTST, class_time = compute_AUCs(XtrRDD, XtstRDD, Bases, Pesos, sigma)

    return AUCTR, AUCTST, ACCTR, ACCTST, class_time, basis_time, pesos_time 


