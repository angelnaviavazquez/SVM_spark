#!/usr/bin/python
# Filename: IRWLSUtils.py

from KernelUtils import *
import time
from scipy.linalg import cho_factor, cho_solve, solve
from sklearn.datasets import load_svmlight_file
from pyspark.mllib.regression import LabeledPoint   
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from SGMAUtils import SGMA, Ballanced_SGMA, SGMAByKey
from ResultsUtils import compute_AUCs, compute_hybrid_AUCs
import numpy as np
from pyspark.mllib.clustering import KMeans, KMeansModel
# IRWLS Procedure


def hybrid_IRWLS(originaldataset,Bases,C,sigma,Niter=100,stop_criteria=1e-6):
    
    #From labeledPoint to tuples label, kernel vector
    nBases=len(Bases)  
    nDims=len(np.array(originaldataset.first().features))

    dataset = originaldataset.map(lambda x: (x.label,np.concatenate((np.array(x.features).reshape((1,nDims)),kncVector(Bases,x.features,sigma).transpose()),axis=1))).cache()
    dataset.count()
    
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
        if condition < stop_criteria:
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


def IRWLSByKey(originaldataset,Bases,C,sigma,Niter=100, stop_criteria=1e-6):
    
    #From labeledPoint to tuples label, kernel vector
    nBases=len(Bases)    
    dataset = originaldataset.map(lambda x: (x.label,kncVector(Bases,x.features,sigma).transpose())).cache()
    dataset.count()
    
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
        (K1,K2) = dataset.map(lambda x: _getK1andK2ByKey(x,Beta,C,i,1.0)).reduceByKey(lambda a,b:(a[0]+b[0],a[1]+b[1])).collect()[0][1]
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
        if condition<stop_criteria:
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


def IRWLS(originaldataset,Bases,C,sigma,Niter=100, stop_criteria=1e-6):
    
    #From labeledPoint to tuples label, kernel vector
    nBases=len(Bases)    
    dataset = originaldataset.map(lambda x: (x.label,kncVector(Bases,x.features,sigma).transpose())).cache()
    #dataset = originaldataset.map(lambda x: (x.label,kncVector(Bases,x.features,sigma).transpose()))
    dataset.count()
    
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
        if condition<stop_criteria:
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


def _getK1andK2ByKey(trainingSet,Beta,C,iteration,samplingRate):

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

    return ('0',resultado)

def loadFile(filename,sc,dimensions, Npartitions):
    X,Y = load_svmlight_file(filename,dimensions)
    X=X.toarray()

    if Npartitions > 0:
        RDD = sc.parallelize(np.concatenate((Y.reshape((len(Y),1)),X),axis=1)).map(lambda x: LabeledPoint(x[0],x[1:]),Npartitions)
    else:
        RDD = sc.parallelize(np.concatenate((Y.reshape((len(Y),1)),X),axis=1)).map(lambda x: LabeledPoint(x[0],x[1:]))

    return RDD


def train_SGMA_IRWLS(XtrRDD, XvalRDD, XtstRDD, sigma, C, NC, Niter=100, stop_criteria=1e-6):

    datasetSize = XtrRDD.count()
    samplingRate=min(1.0,1000.0/datasetSize)

    # comprobando el tipo de etiquetas del dataset, deben ser 0, 1, no -1 , 1
    labels = set(XtrRDD.map(lambda x: x.label).take(100))
    #print labels
    if 0 in labels:
        print "Mapping labels to (-1, 1)..."
        XtrRDD = XtrRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XvalRDD = XvalRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XtstRDD = XtstRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
    time_ini = time.time()

    Bases = SGMA(XtrRDD,NC,sigma,samplingRate)
    
    Pesos = IRWLS(XtrRDD,Bases,C,sigma, stop_criteria=stop_criteria)

    auc_tr, auc_val, auc_tst = compute_AUCs(XtrRDD, XvalRDD, XtstRDD, Bases,Pesos,sigma)

    elapsed_time = time.time() - time_ini
    
    return auc_tr, auc_val, auc_tst, elapsed_time


def train_Ballanced_SGMA_IRWLS(XtrRDD, XvalRDD, XtstRDD, sigma, C, NC, Niter=100, stop_criteria=1e-6):

    datasetSize = XtrRDD.count()
    samplingRate=min(1.0,1000.0/datasetSize)

    # comprobando el tipo de etiquetas del dataset, deben ser 0, 1, no -1 , 1
    labels = set(XtrRDD.map(lambda x: x.label).take(100))
    #print labels
    if 0 in labels:
        print "Mapping labels to (-1, 1)..."
        XtrRDD = XtrRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XvalRDD = XvalRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XtstRDD = XtstRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
    time_ini = time.time()

    Bases = Ballanced_SGMA(XtrRDD,NC,sigma,samplingRate)
    
    Pesos = IRWLS(XtrRDD,Bases,C,sigma, stop_criteria=stop_criteria)

    auc_tr, auc_val, auc_tst = compute_AUCs(XtrRDD, XvalRDD, XtstRDD, Bases,Pesos,sigma)

    elapsed_time = time.time() - time_ini
    
    return auc_tr, auc_val, auc_tst, elapsed_time

def train_SGMA_IRWLS_ByKey(XtrRDD, XvalRDD, XtstRDD, sigma, C, NC, Niter=100, stop_criteria=1e-6):

    datasetSize = XtrRDD.count()
    samplingRate=min(1.0,1000.0/datasetSize)

    # comprobando el tipo de etiquetas del dataset, deben ser 0, 1, no -1 , 1
    labels = set(XtrRDD.map(lambda x: x.label).take(100))
    #print labels
    if 0 in labels:
        print "Mapping labels to (-1, 1)..."
        XtrRDD = XtrRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XvalRDD = XvalRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XtstRDD = XtstRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
    time_ini = time.time()

    Bases = SGMAByKey(XtrRDD,NC,sigma,samplingRate)
    
    Pesos = IRWLSByKey(XtrRDD,Bases,C,sigma, stop_criteria=stop_criteria)

    auc_tr, auc_val, auc_tst = compute_AUCs(XtrRDD, XvalRDD, XtstRDD, Bases,Pesos,sigma)

    elapsed_time = time.time() - time_ini
    
    return auc_tr, auc_val, auc_tst, elapsed_time

def train_hybrid_SGMA_IRWLS(XtrRDD, XvalRDD, XtstRDD, sigma, C, NC, Niter=100, stop_criteria=1e-6):


    # comprobando el tipo de etiquetas del dataset, deben ser 0, 1, no -1 , 1
    labels = set(XtrRDD.map(lambda x: x.label).take(100))
    #print labels
    if 0 in labels:
        print "Mapping labels to (-1, 1)..."
        XtrRDD = XtrRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XvalRDD = XvalRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XtstRDD = XtstRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
    
    datasetSize = XtrRDD.count()
    samplingRate=min(1.0,1000.0/datasetSize)

    # comprobando el tipo de etiquetas del dataset, deben ser 0, 1, no -1 , 1
    labels = set(XtrRDD.map(lambda x: x.label).take(100))
    #print labels
    if 0 in labels:
        print "Mapping labels to (-1, 1)..."
        XtrRDD = XtrRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XvalRDD = XvalRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XtstRDD = XtstRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))

    time_ini = time.time()

    Bases = SGMA(XtrRDD,NC,sigma,samplingRate)
    
    Pesos = hybrid_IRWLS(XtrRDD,Bases,C,sigma, stop_criteria=stop_criteria)

    auc_tr, auc_val, auc_tst = compute_hybrid_AUCs(XtrRDD, XvalRDD, XtstRDD, Bases,Pesos,sigma)
    elapsed_time = time.time() - time_ini
    
    return auc_tr, auc_val, auc_tst, elapsed_time


def train_random_IRWLS(XtrRDD, XvalRDD, XtstRDD, sigma, C, NC, Niter, stop_criteria=1e-6):

    # sustituimos SGMA por random sampling directo

    # comprobando el tipo de etiquetas del dataset, deben ser 0, 1, no -1 , 1
    labels = set(XtrRDD.map(lambda x: x.label).take(100))
    #print labels
    if 0 in labels:
        print "Mapping labels to (-1, 1)..."
        XtrRDD = XtrRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XvalRDD = XvalRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XtstRDD = XtstRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))

    datasetSize = XtrRDD.count()
    samplingRate = min(1.0, 1000.0 / datasetSize)
    
    time_ini = time.time()
    base = XtrRDD.takeSample(False, NC, 1234)
    Bases = [np.array(x.features) for x in base]

    
    Pesos = IRWLS(XtrRDD, Bases, C, sigma, stop_criteria=stop_criteria)

    auc_tr, auc_val, auc_tst = compute_AUCs(XtrRDD, XvalRDD, XtstRDD, Bases, Pesos, sigma)

    elapsed_time = time.time() - time_ini

    return auc_tr, auc_val, auc_tst, elapsed_time


def train_kmeans_IRWLS(XtrRDD, XvalRDD, XtstRDD, sigma, C, NC, Niter, stop_criteria=1e-6):

    # comprobando el tipo de etiquetas del dataset, deben ser 0, 1, no -1 , 1
    labels = set(XtrRDD.map(lambda x: x.label).take(100))
    #print labels
    if 0 in labels:
        print "Mapping labels to (-1, 1)..."
        XtrRDD = XtrRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XvalRDD = XvalRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XtstRDD = XtstRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))

    datasetSize = XtrRDD.count()
    samplingRate = min(1.0, 1000.0 / datasetSize)

    time_ini = time.time()
    
    print "Clustering with Kmeans..."
    clusters = KMeans.train(XtrRDD.map(lambda x: x.features), NC, maxIterations=80, initializationMode="random")
    base = np.array(clusters.centers)
  
    Bases = [np.array(x) for x in base]

   
    Pesos = IRWLS(XtrRDD, Bases, C, sigma, stop_criteria=stop_criteria)

    auc_tr, auc_val, auc_tst = compute_AUCs(XtrRDD, XvalRDD, XtstRDD, Bases, Pesos, sigma)

    elapsed_time = time.time() - time_ini

    return auc_tr, auc_val, auc_tst, elapsed_time
