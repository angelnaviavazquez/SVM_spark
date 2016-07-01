#!/usr/bin/python
# Filename: SGMAUtils.py

from scipy.linalg import cho_solve, solve_triangular
from joblib import Parallel, delayed
import multiprocessing
import time
import sys
import numpy as np
from KernelUtils import *

# It obtains the variable z of the error decrease.

def _get_zeta(KcChol,knc):
    return cho_solve((KcChol,True),knc)


# It obtains the varaible eta of the error decrease.

def _get_eta(zeta,knc):
    return 1.00001-((zeta.transpose().dot(knc))[0,0])

# Partial Error Decrease

def _errorDecrease(dato,Kdato,candidates,sigma,zetalist,samplingRate):
    resultado=np.zeros(len(candidates)) 
    if np.random.random()<samplingRate:
        if len(zetalist)==0:
            for i in range(len(candidates)):
                resultado[i]=(kernel(candidates[i],dato,sigma)**2.0)  
        else:
            for i in range(len(candidates)):
                resultado[i]=((Kdato.dot(zetalist[i])-kernel(candidates[i],dato,sigma))**2.0)        
    return resultado


def _errorDecreaseByKey(dato,Kdato,candidates,sigma,zetalist,samplingRate):
    resultado=np.zeros(len(candidates)) 
    if np.random.random()<samplingRate:
        if len(zetalist)==0:
            for i in range(len(candidates)):
                resultado[i]=(kernel(candidates[i],dato,sigma)**2.0)  
        else:
            for i in range(len(candidates)):
                resultado[i]=((Kdato.dot(zetalist[i])-kernel(candidates[i],dato,sigma))**2.0)        
    return ('0',resultado)


# Rank 1 update of Cholesky Factorization.

def _rankUpdate(KcChol,knc):
    nval = solve_triangular(KcChol, knc,lower=True)
    KcChol = np.concatenate((KcChol,np.zeros(knc.shape)),axis=1)
    row = np.concatenate((nval.transpose(),np.sqrt(np.array([[1.00001]])-nval.transpose().dot(nval))),axis=1)
    return np.concatenate((KcChol,row),axis=0)

# SGMA Function

def SGMAByKey(originaldataset,numCentros,sigma,samplingRate):

    sys.setrecursionlimit(30000)
    
    # Basis list and Cholesky Factorization Initialization
    
    Bases=list()
    KcChol = np.array([[]])
    
    # Mapping from labeledPoint features to numpy array
    
    originaldataset=originaldataset.map(lambda x: np.array(x.features))
    originaldataset.cache()
    
    dataset =  originaldataset.map(lambda x:(x,np.array([])))
    dataset.count()
    
    # Counting the number of cores
    
    num_cores = multiprocessing.cpu_count()
        
    # Loop that takes a new centroid in every iteration

    for cent in range(numCentros):
        
        tInicioIter = time.time()
        
        # Choosing Candidates
        
        print "Centroid",(cent+1),": Taking candidates,",
        candidates=originaldataset.takeSample(True,59)
    
        # Obtaining error decrease
    
        print "Evaluating ED,",
        if len(Bases)==0: 
            
            # First Centroid
            Descenso =  dataset.map(lambda x:_errorDecreaseByKey(x[0],x[1],candidates,sigma,list(),samplingRate)).reduceByKey(lambda a,b:a+b).collect()[0][1]
        else:
            
            # Parallel variables
            knclist = Parallel(n_jobs=num_cores)(delayed(kncVector)(Bases,candidate,sigma) for candidate in candidates)
            zetalist = Parallel(n_jobs=num_cores)(delayed(_get_zeta)(KcChol,knc) for knc in knclist)
            etalist = Parallel(n_jobs=num_cores)(delayed(_get_eta)(zetalist[i],knclist[i]) for i in range(len(candidates)))
    

            # Cluster variables
            Descenso =  dataset.map(lambda x:_errorDecreaseByKey(x[0],x[1],candidates,sigma,zetalist,samplingRate)).reduceByKey(lambda a,b:a+b).collect()[0][1]
            for i in range(len(etalist)):
                Descenso[i]=etalist[i]*Descenso[i]
            
            etalist=np.array(etalist)
            Descenso[etalist<=0.0]=0.0
        
        
        # Updating model
        
        maximo = np.argmax(Descenso)
        print "Max ED:",Descenso[maximo],",",
        
        print "Updating Matrices",
              
        dataset = dataset.map(lambda x:(x[0],np.concatenate((x[1],np.array([kernel(candidates[maximo],x[0],sigma)]))))).cache()

        if len(Bases)==0:
            KcChol=np.sqrt(np.array([[1.00001]]))
        else:
            KcChol=_rankUpdate(KcChol,knclist[maximo])  

        Bases.append(candidates[maximo])    
        
        # Print iteration time
        tFinIter = time.time()
        print "Time", (tFinIter-tInicioIter)  
        
    return Bases        


def SGMA(originaldataset,numCentros,sigma,samplingRate):

    sys.setrecursionlimit(30000)
    
    # Basis list and Cholesky Factorization Initialization
    
    Bases=list()
    KcChol = np.array([[]])
    
    # Mapping from labeledPoint features to numpy array
    
    originaldataset=originaldataset.map(lambda x: np.array(x.features))
    originaldataset.cache()
    
    dataset =  originaldataset.map(lambda x:(x,np.array([])))
    dataset.count()
    
    # Counting the number of cores
    
    num_cores = multiprocessing.cpu_count()
        
    # Loop that takes a new centroid in every iteration

    for cent in range(numCentros):
        
        tInicioIter = time.time()
        
        # Choosing Candidates
        
        print "Centroid",(cent+1),": Taking candidates,",
        candidates=originaldataset.takeSample(True,59)
    
        # Obtaining error decrease
    
        print "Evaluating ED,",
        if len(Bases)==0: 
            
            # First Centroid
            Descenso =  dataset.map(lambda x:_errorDecrease(x[0],x[1],candidates,sigma,list(),samplingRate)).reduce(lambda a,b:a+b)
        else:
            
            # Parallel variables
            knclist = Parallel(n_jobs=num_cores)(delayed(kncVector)(Bases,candidate,sigma) for candidate in candidates)
            zetalist = Parallel(n_jobs=num_cores)(delayed(_get_zeta)(KcChol,knc) for knc in knclist)
            etalist = Parallel(n_jobs=num_cores)(delayed(_get_eta)(zetalist[i],knclist[i]) for i in range(len(candidates)))
    

            # Cluster variables
            Descenso =  dataset.map(lambda x:_errorDecrease(x[0],x[1],candidates,sigma,zetalist,samplingRate)).reduce(lambda a,b:a+b)
            for i in range(len(etalist)):
                Descenso[i]=etalist[i]*Descenso[i]
            
            etalist=np.array(etalist)
            Descenso[etalist<=0.0]=0.0
        
        
        # Updating model
        
        maximo = np.argmax(Descenso)
        print "Max ED:",Descenso[maximo],",",
        
        print "Updating Matrices",
              
        dataset = dataset.map(lambda x:(x[0],np.concatenate((x[1],np.array([kernel(candidates[maximo],x[0],sigma)]))))).cache()

        if len(Bases)==0:
            KcChol=np.sqrt(np.array([[1.00001]]))
        else:
            KcChol=_rankUpdate(KcChol,knclist[maximo])  

        Bases.append(candidates[maximo])    
        
        # Print iteration time
        tFinIter = time.time()
        print "Time", (tFinIter-tInicioIter)  
        
    return Bases        


def Ballanced_SGMA(originaldataset,numCentros,sigma,samplingRate):

    sys.setrecursionlimit(30000)
    
    # Basis list and Cholesky Factorization Initialization
    
    Bases=list()
    KcChol = np.array([[]])
    
    # Mapping from labeledPoint features to numpy array
    
    positiveDataset=originaldataset.filter(lambda x: x.label==1.0).map(lambda x: np.array(x.features)).cache()
    
    negativeDataset=originaldataset.filter(lambda x: x.label!=1.0).map(lambda x: np.array(x.features)).cache()
    
    dataset =  originaldataset.map(lambda x:(np.array(x.features),np.array([]))).cache()
    dataset.count()
    
    # Counting the number of cores
    
    num_cores = multiprocessing.cpu_count()
        
    # Loop that takes a new centroid in every iteration

    for cent in range(numCentros):
        
        tInicioIter = time.time()
        
        # Choosing Candidates
        
        print "Centroid",(cent+1),": Taking candidates,",
        
        if cent % 2 == 0:
            candidates=positiveDataset.takeSample(True,59)
        else:
            candidates=negativeDataset.takeSample(True,59)
    
        # Obtaining error decrease
    
        print "Evaluating ED,",
        if len(Bases)==0: 
            
            # First Centroid
            Descenso =  dataset.map(lambda x:_errorDecrease(x[0],x[1],candidates,sigma,list(),samplingRate)).reduce(lambda a,b:a+b)
        else:
            
            # Parallel variables
            knclist = Parallel(n_jobs=num_cores)(delayed(kncVector)(Bases,candidate,sigma) for candidate in candidates)
            zetalist = Parallel(n_jobs=num_cores)(delayed(_get_zeta)(KcChol,knc) for knc in knclist)
            etalist = Parallel(n_jobs=num_cores)(delayed(_get_eta)(zetalist[i],knclist[i]) for i in range(len(candidates)))
    

            # Cluster variables
            Descenso =  dataset.map(lambda x:_errorDecrease(x[0],x[1],candidates,sigma,zetalist,samplingRate)).reduce(lambda a,b:a+b)
            for i in range(len(etalist)):
                Descenso[i]=etalist[i]*Descenso[i]
            
            etalist=np.array(etalist)
            Descenso[etalist<=0.0]=0.0
        
        
        # Updating model
        
        maximo = np.argmax(Descenso)
        print "Max ED:",Descenso[maximo],",",
        
        print "Updating Matrices",
              
        dataset = dataset.map(lambda x:(x[0],np.concatenate((x[1],np.array([kernel(candidates[maximo],x[0],sigma)]))))).cache()

        if len(Bases)==0:
            KcChol=np.sqrt(np.array([[1.00001]]))
        else:
            KcChol=_rankUpdate(KcChol,knclist[maximo])  

        Bases.append(candidates[maximo])    
        
        # Print iteration time
        tFinIter = time.time()
        print "Time", (tFinIter-tInicioIter)  
        
    return Bases        
