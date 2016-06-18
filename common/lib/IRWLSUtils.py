#!/usr/bin/python
# Filename: IRWLSUtils.py

from KernelUtils import *
import time
from scipy.linalg import cho_factor, cho_solve, solve
from sklearn.datasets import load_svmlight_file
from pyspark.mllib.regression import LabeledPoint   
from SGMAUtils import SGMA
from ResultsUtils import compute_AUCs, compute_hybrid_AUCs
import numpy as np
# IRWLS Procedure


def hybrid_IRWLS(originaldataset,Bases,C,sigma,Niter=100):
    
    #From labeledPoint to tuples label, kernel vector
    nBases=len(Bases)  
    nDims=len(np.array(originaldataset.first().features))

    dataset = originaldataset.map(lambda x: (x.label,np.concatenate((np.array(x.features).reshape((1,nDims)),kncVector(Bases,x.features,sigma).transpose()),axis=1)))
    
    # Basis kernel matrix
    KC=np.zeros((nDims+nBases,nDims+nBases))
    KC[0:nDims,0:nDims]=np.diag(np.ones(nDims))
    KC[nDims:,nDims:]=kernelMatrix(Bases,Bases,sigma)+np.diag(1e-6*np.ones(len(Bases)))
    
    # Weights Initialization
    Beta = np.zeros(nBases)
    bestBeta = Beta    
    iterSinceBestCondition = 0
    bestCondition = np.Infinity    
    
    for i in range(Niter):
        
        tInicioIter = time.time()
        
        # IRWLS Procedure
        (K1,K2) = dataset.map(lambda x: _getK1andK2(x,Beta,C,i,1.0)).reduce(lambda a,b:(a[0]+b[0],a[1]+b[1]))
        K1[0:nBases+nDims,0:nBases+nDims]=K1[0:nBases+nDims,0:nBases+nDims]+KC 
                
        newBeta = solve(K1,K2)
        #try:               
        #    K1Chol = cho_factor(K1)
        #    newBeta = cho_solve(K1Chol,K2)
        #except Exception as inst:
        #    return  bestBeta

        # Convergence criteria
        condition = np.linalg.norm(Beta-newBeta)/np.linalg.norm(Beta)                
        Beta=newBeta

        # Check convergence
        if condition<1e-6:
            return bestBeta
        
        if condition<bestCondition:
            iterSinceBestCondition=0
            bestCondition=condition
            bestBeta=Beta
        else:
            iterSinceBestCondition+=1                
            if iterSinceBestCondition>=5:
                return bestBeta
        
        tFinIter = time.time()
        
        print "Iteration",(i+1),": DeltaW/W",condition,", Iteration Time", (tFinIter-tInicioIter)
        
    return bestBeta   


def IRWLS(originaldataset,Bases,C,sigma,Niter=100):
    
    #From labeledPoint to tuples label, kernel vector
    nBases=len(Bases)    
    dataset = originaldataset.map(lambda x: (x.label,kncVector(Bases,x.features,sigma).transpose()))
    
    # Basis kernel matrix
    KC=kernelMatrix(Bases,Bases,sigma)
    
    # Weights Initialization
    Beta = np.zeros(nBases)
    bestBeta = Beta    
    iterSinceBestCondition = 0
    bestCondition = np.Infinity    
    
    for i in range(Niter):
        
        tInicioIter = time.time()
        
        # IRWLS Procedure
        (K1,K2) = dataset.map(lambda x: _getK1andK2(x,Beta,C,i,1.0)).reduce(lambda a,b:(a[0]+b[0],a[1]+b[1]))
        K1[0:nBases,0:nBases]=K1[0:nBases,0:nBases]+KC   

        newBeta = solve(K1,K2)
        #try:               
        #    K1Chol = cho_factor(K1)
        #    newBeta = cho_solve(K1Chol,K2)
        #except Exception as inst:
        #    return  bestBeta

        # Convergence criteria
        condition = np.linalg.norm(Beta-newBeta)/np.linalg.norm(Beta)                
        Beta=newBeta

        # Check convergence
        if condition<1e-6:
            return bestBeta
        
        if condition<bestCondition:
            iterSinceBestCondition=0
            bestCondition=condition
            bestBeta=Beta
        else:
            iterSinceBestCondition+=1                
            if iterSinceBestCondition>=5:
                return bestBeta

        
        tFinIter = time.time()
        
        print "Iteration",(i+1),": DeltaW/W",condition,", Iteration Time", (tFinIter-tInicioIter)
        
    return bestBeta    


# Stochastic IRWLS version

def StochasticIRWLS(originaldataset, Bases, C, sigma, Niter, samplingRate, eta):
    
    #From labeledPoint to tuples label, kernel vector
    nBases=len(Bases)    
    dataset = originaldataset.map(lambda x: (x.label,kncVector(Bases,x.features,sigma).transpose()))
    
    # Basis kernel matrix
    KC=kernelMatrix(Bases,Bases,sigma)
    
    # Weights Initizalization
    Beta = np.zeros(nBases)
            
    for i in range(Niter):        
  
        tInicioIter = time.time()
    
        # IRWLS procedure
        (K1,K2) = dataset.map(lambda x: _getK1andK2(x,Beta,C,0,samplingRate)).reduce(lambda a,b:(a[0]+b[0],a[1]+b[1]))
        K1[0:nBases,0:nBases]=K1[0:nBases,0:nBases]+KC                
        K1Chol = cho_factor(K1)
        newBeta = cho_solve(K1Chol,K2)

        # Weights Updating
        Beta = eta*newBeta + (1.0-eta)*Beta
        
        tFinIter = time.time()
        
        print "Iteration",(i+1),": Iteration Time", (tFinIter-tInicioIter)
        
    return Beta    


def _getK1andK2(trainingSet,Beta,C,iteration,samplingRate):

    resultado = (0.0,0.0)

    if np.random.random()<samplingRate:
        if iteration == 0:
            a=1
        else:
            error = trainingSet[0]-trainingSet[1].dot(Beta)[0,0]
            a=C/(samplingRate*error*trainingSet[0])   
            if a>10000.0:
                a=10000.0
                
        if a>0.0:
            resultado = (a*trainingSet[1].transpose().dot(trainingSet[1]),a*trainingSet[1].transpose().dot(trainingSet[0]))

    return resultado

def loadFile(filename,sc,dimensions, Npartitions):
    X,Y = load_svmlight_file(filename,dimensions)
    X=X.toarray()
    return sc.parallelize(np.concatenate((Y.reshape((len(Y),1)),X),axis=1)).map(lambda x: LabeledPoint(x[0],x[1:]),Npartitions)


def train_SGMA_IRWLS(XtrRDD, XvalRDD, XtstRDD, sigma, C, NC, Niter):

    time_ini = time.time()
    gamma = 1.0/(sigma*sigma)   
    datasetSize = XtrRDD.count()
    samplingRate=min(1.0,1000.0/datasetSize)

    Bases = SGMA(XtrRDD,NC,sigma,samplingRate)
    
    Pesos = IRWLS(XtrRDD,Bases,C,sigma)

    auc_tr, auc_val, auc_tst = compute_AUCs(XtrRDD, XvalRDD, XtstRDD, Bases,Pesos,sigma)

    elapsed_time = time.time() - time_ini
    
    print "AUCtr = %f, AUCval = %f, AUCtst = %f" % (auc_tr, auc_val, auc_tst)
    print "Elapsed_time = %f" % elapsed_time
    return auc_tr, auc_val, auc_tst, elapsed_time


def train_hybrid_SGMA_IRWLS(XtrRDD, XvalRDD, XtstRDD, sigma, C, NC, Niter):

    time_ini = time.time()
    
    datasetSize = XtrRDD.count()
    samplingRate=min(1.0,1000.0/datasetSize)

    Bases = SGMA(XtrRDD,NC,sigma,samplingRate)
    
    Pesos = hybrid_IRWLS(XtrRDD,Bases,C,sigma)

    auc_tr, auc_val, auc_tst = compute_hybrid_AUCs(XtrRDD, XvalRDD, XtstRDD, Bases,Pesos,sigma)
    elapsed_time = time.time() - time_ini
    
    print "AUCtr = %f, AUCval = %f, AUCtst = %f" % (auc_tr, auc_val, auc_tst)
    print "Elapsed_time = %f" % elapsed_time
    return auc_tr, auc_val, auc_tst, elapsed_time


def train_random_IRWLS(XtrRDD, XvalRDD, XtstRDD, sigma, C, NC, Niter):

    # sustituimos SGMA por random sampling directo
    time_ini = time.time()
    gamma = 1.0 / (sigma * sigma)

    datasetSize = XtrRDD.count()
    samplingRate = min(1.0, 1000.0 / datasetSize)
    
    base = XtrRDD.takeSample(False, NC, 1234)
    Bases = [np.array(x.features) for x in base]

    Pesos = IRWLS(XtrRDD, Bases, C, gamma)

    auc_tr, auc_val, auc_tst = compute_AUCs(XtrRDD, XvalRDD, XtstRDD, Bases, Pesos, gamma)

    elapsed_time = time.time() - time_ini
    
    print "AUCtr = %f, AUCval = %f, AUCtst = %f" % (auc_tr, auc_val, auc_tst)
    print "Elapsed_time = %f" % elapsed_time
    return auc_tr, auc_val, auc_tst, elapsed_time
