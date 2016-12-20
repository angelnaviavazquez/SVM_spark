#!/usr/bin/python
##  Functions to train the Distributed Semiparametric SVM
#
#  A distributed stochastic version of the Sparse Greedy Matrix Approximation algorithm

from KernelUtils import *
import time
from scipy.linalg import solve
from sklearn.datasets import load_svmlight_file
from pyspark.mllib.regression import LabeledPoint   
from SGMAUtils import DSSGMA
from ResultsUtils import compute_AUCs
import numpy as np

## Distributed Iterative Re-Weighted Least Squares procedure to obtain the weights of a semiparametric model.
#  @param trainingSet The training set
#  @param Beta The current weights of the model
#  @param C The parameter of the cost function
#  @param Niter Maximum number of iterations.
#  @param stop_criteria Stop criteria to finish the IRWLS procedure.

def DIRWLS(originaldataset,Bases,C,sigma,Niter=100, stop_criteria=1e-3):
    
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
    NSVs = -99
    
    for i in range(Niter):
        
        tInicioIter = time.time()
        
        # IRWLS Procedure
        (K1,K2) = dataset.map(lambda x: _getK1andK2(x,Beta,C,i,1.0)).reduce(lambda a,b:(a[0]+b[0],a[1]+b[1]))
        K1[0:nBases,0:nBases]=K1[0:nBases,0:nBases]+KC   

        newBeta = solve(K1,K2)

        # Convergence criteria
        condition = np.linalg.norm(Beta-newBeta)/np.linalg.norm(Beta)                
        Beta=newBeta

        # Check convergence
        if condition<stop_criteria:
            break
        
        if condition<bestCondition:
            iterSinceBestCondition=0
            bestCondition=condition
            bestBeta=Beta
        else:
            iterSinceBestCondition+=1                
            if iterSinceBestCondition>=5:
                break
        
        tFinIter = time.time()
        
        print "Iteration",(i+1),": DeltaW/W",condition,", Iteration Time", (tFinIter-tInicioIter)    
        
    nSVs = dataset.map(lambda x: _isSV(x,Beta,C)).reduce(lambda a,b:a+b)
    
    return bestBeta, nSVs

## Function to detect if a sample is a Support Vector
#  @param trainingSet The training set
#  @param Beta The current weights of the model
#  @param C The parameter of the cost function

def _isSV(trainingSet,Beta,C):
    
    error = trainingSet[0]-trainingSet[1].dot(Beta)[0,0]
    a=C/(error*trainingSet[0]) 
    
    if a>0.0:
        return 1
    else:
        return 0

## Auxiliar function to obtain the matrix for the weighted least squares problem in the IRWLS procedure.
#  @param trainingSet The training set
#  @param Beta The current weights of the model
#  @param C The parameter of the cost function
#  @param iteration The iteration number.
#  @param samplingRate The percentaje of training data to use.

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


## Function to load a dataset in libsvm format and distribute them among the nodes of the Spark Cluster.
#  @param filename The file name.
#  @param sc The spark context.
#  @param dimensions Number of features.
#  @param Npartitions The number of partitions to divide the dataset.

def loadFile(filename,sc,dimensions, Npartitions):
    X,Y = load_svmlight_file(filename,dimensions)
    X=X.toarray()

    if Npartitions > 0:
        RDD = sc.parallelize(np.concatenate((Y.reshape((len(Y),1)),X),axis=1)).map(lambda x: LabeledPoint(x[0],x[1:]),Npartitions)
    else:
        RDD = sc.parallelize(np.concatenate((Y.reshape((len(Y),1)),X),axis=1)).map(lambda x: LabeledPoint(x[0],x[1:]))

    return RDD


## Complete function to train a distributed semiparametric SVM. It makes use of the DSSGMA algorithm to obtain the basis elements and the DIRWLS procedure to obtain the weights.
#  @param XtrRDD Distributed training dataset.
#  @param XvalRDD Distributed Validation dataset.
#  @param XtstRDD Distributed test dataset.
#  @param sigma  The kernel parameter
#  @param C The parameter of the cost function
#  @param NC Final machine size.
#  @param Niter Maximum number of iterations.
#  @param stop_criteria Stop criteria to finish the IRWLS procedure.

def train_SGMA_IRWLS(XtrRDD, XvalRDD, XtstRDD, sigma, C, NC, Niter=100, stop_criteria=1e-3):

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

    Bases = DSSGMA(XtrRDD,NC,sigma,samplingRate)
    
    Pesos, NSVs = DIRWLS(XtrRDD,Bases,C,sigma, stop_criteria=stop_criteria)

    auc_tr, auc_val, auc_tst = compute_AUCs(XtrRDD, XvalRDD, XtstRDD, Bases,Pesos,sigma)

    elapsed_time = time.time() - time_ini
    
    return auc_tr, auc_val, auc_tst, elapsed_time



