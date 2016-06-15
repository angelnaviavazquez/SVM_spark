#!/usr/bin/python
# Filename: IRWLSUtils.py

from KernelUtils import *
import time
from scipy.linalg import cho_factor, cho_solve
from sklearn.datasets import load_svmlight_file
from pyspark.mllib.regression import LabeledPoint   
from SGMAUtils import SGMA
from ResultsUtils import compute_AUCs
# IRWLS Procedure

def IRWLS(originaldataset,Bases,C,gamma,Niter=100):
    
    #From labeledPoint to tuples label, kernel vector
    nBases=len(Bases)    
    dataset = originaldataset.map(lambda x: (x.label,kncVector(Bases,x.features,gamma).transpose()))
    
    # Basis kernel matrix
    KC=kernelMatrix(Bases,Bases,gamma)
    
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

        try:               
            K1Chol = cho_factor(K1)
            newBeta = cho_solve(K1Chol,K2)
        except Exception as inst:
            return  bestBeta

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

def StochasticIRWLS(originaldataset, Bases, C, gamma, Niter, samplingRate, eta):
    
    #From labeledPoint to tuples label, kernel vector
    nBases=len(Bases)    
    dataset = originaldataset.map(lambda x: (x.label,kncVector(Bases,x.features,gamma).transpose()))
    
    # Basis kernel matrix
    KC=kernelMatrix(Bases,Bases,gamma)
    
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
        
        if a>0.0:
            resultado = (a*trainingSet[1].transpose().dot(trainingSet[1]),a*trainingSet[1].transpose().dot(trainingSet[0]))

    return resultado

def loadFile(filename,sc,dimensions):
    X,Y = load_svmlight_file(filename,dimensions)
    X=X.toarray()
    return sc.parallelize(np.concatenate((Y.reshape((len(Y),1)),X),axis=1)).map(lambda x: LabeledPoint(x[0],x[1:]),12)


def train_SGMA_IRWLS(XtrRDD, XvalRDD, XtstRDD, sigma, C, NC, name_dataset, Niter, Samplefraction):

    time_ini = time.time()
    gamma = 1.0/(sigma*sigma)
    datasetSize = XtrRDD.count()
    samplingRate=min(1.0,1000.0/datasetSize)

    Bases = SGMA(XtrRDD,NC,gamma,samplingRate)
    
    Pesos = IRWLS(XtrRDD,Bases,C,gamma)

    auc_val, auc_tst = compute_AUCs(XvalRDD, XtstRDD, Bases,Pesos,gamma)

    exe_time = time.time() - time_ini
    print "AUCval = %f, AUCtst = %f" % (auc_val, auc_tst)
    print "Elapsed_time = %f" % exe_time
 
    return auc_val, auc_tst, exe_time
