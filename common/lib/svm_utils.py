# -*- coding: utf-8 -*-
import numpy as np
from pyspark.mllib.regression import LabeledPoint
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import scipy.io
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.clustering import GaussianMixture
from numpy import array
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from time import time
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel

'''
Created on 13/06/2016
@author: angel.navia@uc3m.es

import code
code.interact(local=locals())
'''


def build_X1(x):
    NI = len(x.features)
    k = np.vstack((np.array(1).reshape((1, 1)), x.features.reshape((NI, 1))))
    return LabeledPoint(x.label, k.T)


def load_data(kdataset, kfold):
    if kdataset == 0:
        mat = scipy.io.loadmat('./data/ripley_5fold.mat')
    if kdataset == 1:
        mat = scipy.io.loadmat('./data/kwok_5fold.mat')
    if kdataset == 2:
        mat = scipy.io.loadmat('./data/twonorm_5fold.mat')
    if kdataset == 3:
        mat = scipy.io.loadmat('./data/waveform_5fold.mat')

    index_tr = mat['index_tr']
    index_val = mat['index_val']
    x_tr = mat['x_tr']
    x_tst = mat['x_tst']
    y_tr = mat['y_tr']
    y_tst = mat['y_tst']

    ind_tr = np.where(index_tr[:, kfold] == 1)
    ind_val = np.where(index_val[:, kfold] == 1)

    x_tr_ = x_tr[ind_tr]
    y_tr_ = y_tr[ind_tr]

    x_val_ = x_tr[ind_val]
    y_val_ = y_tr[ind_val]

    x_tst_ = x_tst
    y_tst_ = y_tst

    return x_tr_, y_tr_, x_val_, y_val_, x_tst_, y_tst_


def get_inc_w(x, w, landa):
    k = x.features
    NI = len(k)
    k = k.reshape(NI, 1)
    ytr = x.label
    yKw = ytr * np.dot(k.T, w)
    if yKw < 1:
        incw = -landa * w + ytr * k
    else:
        incw = -landa * w
    return incw


def train_linearSVM(Xtr1RDD, NI, C, eta, landa, Niter, Samplefraction):
    w = np.zeros(NI + 1).reshape((NI + 1, 1))
    N_added = 0
    for k in range(0, Niter):
        print k,
        Xtr1RDDsampled = Xtr1RDD.sample(withReplacement=True, fraction=Samplefraction)
        incwRDD = Xtr1RDDsampled.map(lambda x: get_inc_w(x, w, landa))
        Nsample = incwRDD.count()
        N_added += Nsample
        eta = C / N_added
        w += eta * incwRDD.reduce(lambda x, y: x + y)
    return w


def in_margin(x, w):
    k = x.features
    ytr = x.label
    yKw = ytr * np.dot(k, w)
    return yKw < 1


def plot_linear_SVM(xtr, ytr, w, c, name_dataset):
    
    index1labels = np.where(ytr > 0)
    index0labels = np.where(ytr <= 0)

    plt.plot(xtr[index1labels, 0], xtr[index1labels, 1], 'b.')
    plt.plot(xtr[index0labels, 0], xtr[index0labels, 1], 'r.')
    NP = 100.0
    SUP = np.zeros((NP, NP))
    ejex = np.arange(NP).astype(float) / NP * 4.0 - 2.0
    X_mesh = np.array([0, 0, 0]).reshape((1,3))
    for m in range(0, int(NP)):
        for n in range(0, int(NP)):
            aux = np.array([1.0, ejex[m], ejex[n]]).reshape((1,3))
            X_mesh = np.vstack((X_mesh, aux))

    y_pred = np.dot(X_mesh, w)
    y_pred = y_pred[1:]
    Z = y_pred.reshape((NP,NP))
    levels = np.arange(-1.0, 2.0 , 1.0)

    CS = plt.contour(ejex, ejex, Z.T, levels, colors = 'k')
    plt.clabel(CS, fontsize=9, inline=1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(name_dataset)
    plt.grid(True)
    #plt.savefig("test.png")
    plt.plot(c[:, 0], c[:, 1], 'ro')
    plt.show()

def plot_hybrid_SVM(xtr, ytr, w, c, name_dataset):
    index1labels = np.where(ytr > 0)
    index0labels = np.where(ytr <= 0)
    plt.plot(xtr[index1labels,0], xtr[index1labels,1],'b.')
    plt.plot(xtr[index0labels,0], xtr[index0labels,1],'r.')
    NI_a = xtr.shape[1]
    NI_b = c.shape[0]
    NI = NI_a + NI_b + 1
    NP = 100.0
    SUP = np.zeros((NP,NP))
    ejex = np.arange(NP).astype(float) / NP * 4.0 - 2.0
    X_mesh = np.zeros(NI).reshape((1,NI))
    
    NC = c.shape[0]
    for m in range(0, int(NP)):
        for n in range(0, int(NP)):
            x1 = np.array([1.0, ejex[m], ejex[n]]).reshape((1,3))
            X = np.kron(np.ones(NC).reshape((NC,1)),[ejex[m], ejex[n]])
            e = X-c
            e = e**2
            e = e.sum(axis=1)
            k = np.exp(-0.5 * e /sigma/sigma)
            k = k.reshape((1, NC))
            kx1 = np.hstack((x1,k))
            X_mesh = np.vstack((X_mesh, kx1))

    y_pred = np.dot(X_mesh, w)
    y_pred = y_pred[1:]
    Z = y_pred.reshape((NP,NP))
    levels = np.arange(-1.0, 2.0 , 1.0)
    CS = plt.contour(ejex, ejex, Z.T, levels, colors = 'k')
    plt.clabel(CS, fontsize=9, inline=1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(name_dataset)
    plt.grid(True)
    plt.show()

    
def kernelG(x,c,sigma):
    NC = c.shape[0]
    X = np.kron(np.ones(NC).reshape((NC,1)),x.features)
    e = X-c
    e = e**2
    e = e.sum(axis=1)
    k = np.exp(-0.5 * e /sigma/sigma)
    k = k.reshape((NC, 1))
    return k

def build_k(x, c, sigma):
    NI = len(x.features)
    k = np.vstack((np.array(1).reshape((1,1)),x.features.reshape((NI, 1))))
    k = np.vstack((k,kernelG(x,c,sigma)))
    x = LabeledPoint(x.label, k.T)
    return x

def build_kc(x, c, sigma):
    #NI = len(x.features)
    #k = np.vstack((np.array(1).reshape((1,1)),x.features.reshape((NI, 1))))
    #import code
    #code.interact(local=locals())

    k = np.vstack((np.array(1).reshape((1,1)),kernelG(x,c,sigma)))
    x = LabeledPoint(x.label, k.T)
    return x

def train_nonlinearSVM(KtrRDD, C, landa, Niter, Samplefraction):
    x = KtrRDD.take(1)[0]
    NI = len(x.features)
    w = np.zeros(NI).reshape((NI,1))
    N_added = 0
    #KtrRDDsampled = KtrRDD.sample(withReplacement=True, fraction=Samplefraction)

    for k in range(0,Niter):
        print k, 
        KtrRDDsampled = KtrRDD.sample(withReplacement=True, fraction=Samplefraction)
        incwRDD = KtrRDDsampled.map(lambda x: get_inc_w(x, w, landa))
        Nsample = incwRDD.count()
        N_added += Nsample 
        eta = C/N_added
        deltaw = eta * incwRDD.reduce(lambda x, y: x + y)
        w += deltaw
        #print np.linalg.norm(deltaw)
    print "Fin!"
    return w


def predict(x, w):
    k = x.features.reshape((1,len(x.features)))
    y_pred = np.dot(k, w)
    return y_pred

def plot_ROC(Ytr, Ytst):
    fpr_tr, tpr_tr, th_tr = roc_curve(np.array(Ytr)[:,0], np.array(Ytr)[:,1])
    auc_tr = auc(fpr_tr, tpr_tr)
    #plt.plot(fpr_tr, tpr_tr,'r')

    fpr_tst, tpr_tst, th_tst = roc_curve(np.array(Ytst)[:,0], np.array(Ytst)[:,1])
    auc_tst = auc(fpr_tst, tpr_tst)
    #plt.plot(fpr_tst, tpr_tst,'g')

    '''
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(name_dataset)
    plt.grid(True)
    plt.show()
    '''
    return auc_tst

                  
def train_kernelgrad(XtrRDD, XvalRDD, XtstRDD, sigma, C, NC, Niter, Samplefraction):
    
    time_ini = time()

    # comprobando el tipo de etiquetas del dataset, deben ser 0, 1, no -1 , 1
    labels = set(XtrRDD.map(lambda x: x.label).take(100))
    #print labels
    if 0 in labels:
        print "Mapping labels to (-1, 1)..."
        XtrRDD = XtrRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XvalRDD = XvalRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XtstRDD = XtstRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))

    eta = C
    landa = 1.0 / C
    NPtr = XtrRDD.count()
    #NPval = XvalRDD.count()
    #NPtst = XtstRDD.count()
    NI = len(XtrRDD.take(1)[0].features)
    #XtrRDD.cache()
    #XtstRDD.cache()
    T = NPtr
    
    #print NPtr, NPtst, NI

    #print "Training the linear SVM model during %d iterations" % Niter

    Xtr1RDD = XtrRDD.map(lambda x: build_X1(x)).cache()
    #Xtst1RDD = XtstRDD.map(lambda x: build_X1(x)).cache()
    #w = train_linearSVM(Xtr1RDD, NI, C, eta, landa, Niter, Samplefraction)
    #print "Done!"

    #xtr = np.array(XtrRDD.map(lambda x: x.features).collect())
    #ytr = np.array(XtrRDD.map(lambda x: x.label).collect())

    print "Clustering data..."
    #SV_RDD = Xtr1RDD.filter(lambda x: in_margin(x, w))
    clusters = KMeans.train(Xtr1RDD.map(lambda x: x.features[1:len(x.features)]), NC, maxIterations=20, runs=20, initializationMode="random")
    c = np.array(clusters.centers)

    '''
    if kdataset == 1 or kdataset == 2:   # el resto no se pueden pintar
        #SVM.plot_linear_SVM(xtr, ytr, w, c)
        plot_linear_SVM(xtr, ytr, w, c, name_dataset)
    '''
    
    print "Building the kernel expansion..."    
    KtrRDD = XtrRDD.map(lambda x: build_kc(x, c, sigma)).cache()
    KvalRDD = XvalRDD.map(lambda x: build_kc(x, c, sigma)).cache()
    KtstRDD = XtstRDD.map(lambda x: build_kc(x, c, sigma)).cache()

    print "Training the hybrid SVM model during %d iterations" % Niter

    w = train_nonlinearSVM(KtrRDD, C, landa, Niter, Samplefraction)

    '''
    xtr = np.array(XtrRDD.map(lambda x: x.features).collect())
    ytr = np.array(XtrRDD.map(lambda x: x.label).collect())
    if kdataset == 1 or kdataset == 2:   # el resto no se pueden pintar 
        plot_hybrid_SVM(xtr, ytr, w, c, name_dataset)
    '''
    print "Predicting and evaluating..."

    y_pred_trRDD = KtrRDD.map(lambda x: (x.label, predict(x, w)[0][0]))
    y_pred_valRDD = KvalRDD.map(lambda x: (x.label, predict(x, w)[0][0]))
    y_pred_tstRDD = KtstRDD.map(lambda x: (x.label, predict(x, w)[0][0]))

    Ytr = y_pred_trRDD.collect()
    Yval = y_pred_valRDD.collect()
    Ytst = y_pred_tstRDD.collect()

    #auc_tst = plot_ROC(Ytr, Ytst)
    elapsed_time = time() - time_ini

    fpr_tr, tpr_tr, th_tr = roc_curve(np.array(Ytr)[:,0], np.array(Ytr)[:,1])
    auc_tr = auc(fpr_tr, tpr_tr)

    fpr_val, tpr_val, th_val = roc_curve(np.array(Yval)[:,0], np.array(Yval)[:,1])
    auc_val = auc(fpr_val, tpr_val)

    fpr_tst, tpr_tst, th_tst = roc_curve(np.array(Ytst)[:,0], np.array(Ytst)[:,1])
    auc_tst = auc(fpr_tst, tpr_tst)

    return auc_tr, auc_val, auc_tst, elapsed_time


def train_hybridSVM(XtrRDD, XvalRDD, XtstRDD, sigma, C, NC, Niter, Samplefraction):
    
    time_ini = time()
    eta = C
    landa = 1.0 / C

    # comprobando el tipo de etiquetas del dataset, deben ser 0, 1, no -1 , 1
    labels = set(XtrRDD.map(lambda x: x.label).take(100))
    #print labels
    if 0 in labels:
        print "Mapping labels to (-1, 1)..."
        XtrRDD = XtrRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XvalRDD = XvalRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
        XtstRDD = XtstRDD.map(lambda x: LabeledPoint(x.label * 2.0 - 1.0, x.features))
    
    NPtr = XtrRDD.count()
    #NPval = XvalRDD.count()
    #NPtst = XtstRDD.count()
    NI = len(XtrRDD.take(1)[0].features)
    #XtrRDD.cache()
    #XtstRDD.cache()
    T = NPtr
    
    #print NPtr, NPtst, NI

    print "Training the linear SVM model during %d iterations" % Niter
    #import code
    #code.interact(local=locals())

    Xtr1RDD = XtrRDD.map(lambda x: build_X1(x)).cache()
    #Xtst1RDD = XtstRDD.map(lambda x: build_X1(x)).cache()

    w = train_linearSVM(Xtr1RDD, NI, C, eta, landa, Niter, Samplefraction)
    print "Done!"

    xtr = np.array(XtrRDD.map(lambda x: x.features).collect())
    ytr = np.array(XtrRDD.map(lambda x: x.label).collect())

    print "Clustering SVs..."
    SV_RDD = Xtr1RDD.filter(lambda x: in_margin(x, w))
    clusters = KMeans.train(SV_RDD.map(lambda x: x.features[1:len(x.features)]), NC, maxIterations=80, initializationMode="random")
    c = np.array(clusters.centers)
   
    print "Building the kernel expansion..."    
    KtrRDD = XtrRDD.map(lambda x: build_k(x, c, sigma)).cache()
    KvalRDD = XvalRDD.map(lambda x: build_k(x, c, sigma)).cache()
    KtstRDD = XtstRDD.map(lambda x: build_k(x, c, sigma)).cache()

    print "Training the hybrid SVM model during %d iterations" % Niter

    w = train_nonlinearSVM(KtrRDD, C, landa, Niter, Samplefraction)

    print "Predicting and evaluating..."

    y_pred_trRDD = KtrRDD.map(lambda x: (x.label, predict(x, w)[0][0]))
    y_pred_valRDD = KvalRDD.map(lambda x: (x.label, predict(x, w)[0][0]))
    y_pred_tstRDD = KtstRDD.map(lambda x: (x.label, predict(x, w)[0][0]))

    Ytr = y_pred_trRDD.collect()
    Yval = y_pred_valRDD.collect()
    Ytst = y_pred_tstRDD.collect()

    #auc_tst = plot_ROC(Ytr, Ytst)
    elapsed_time = time() - time_ini

    fpr_tr, tpr_tr, th_tr = roc_curve(np.array(Ytr)[:,0], np.array(Ytr)[:,1])
    auc_tr = auc(fpr_tr, tpr_tr)

    fpr_val, tpr_val, th_val = roc_curve(np.array(Yval)[:,0], np.array(Yval)[:,1])
    auc_val = auc(fpr_val, tpr_val)

    fpr_tst, tpr_tst, th_tst = roc_curve(np.array(Ytst)[:,0], np.array(Ytst)[:,1])
    auc_tst = auc(fpr_tst, tpr_tst)

    return auc_tr, auc_val, auc_tst, elapsed_time
    

def train_linear_SVM(XtrRDD, XvalRDD, XtstRDD):
    time_ini = time()

    # comprobando el tipo de etiquetas del dataset, deben ser 0, 1, no -1 , 1
    labels = set(XtrRDD.map(lambda x: x.label).take(100))
    #print labels
    if -1 in labels:
        XtrRDD = XtrRDD.map(lambda x: LabeledPoint((x.label + 1) / 2, x.features))
        XvalRDD = XvalRDD.map(lambda x: LabeledPoint((x.label + 1) / 2, x.features))
        XtstRDD = XtstRDD.map(lambda x: LabeledPoint((x.label + 1) / 2, x.features))
    #labels = set(XtrRDD.map(lambda x: x.label).take(100))
    #print labels

    model = SVMWithSGD.train(XtrRDD, iterations=100)

    y_pred_trRDD = XtrRDD.map(lambda x: (x.label, model.predict(x.features)))
    y_pred_valRDD = XvalRDD.map(lambda x: (x.label, model.predict(x.features)))
    y_pred_tstRDD = XtstRDD.map(lambda x: (x.label, model.predict(x.features)))

    elapsed_time = time() - time_ini

    Ytr = y_pred_trRDD.collect()
    Yval = y_pred_valRDD.collect()
    Ytst = y_pred_tstRDD.collect()

    fpr_tr, tpr_tr, th_tr = roc_curve(np.array(Ytr)[:,0], np.array(Ytr)[:,1])
    auc_tr = auc(fpr_tr, tpr_tr)

    fpr_val, tpr_val, th_val = roc_curve(np.array(Yval)[:,0], np.array(Yval)[:,1])
    auc_val = auc(fpr_val, tpr_val)

    fpr_tst, tpr_tst, th_tst = roc_curve(np.array(Ytst)[:,0], np.array(Ytst)[:,1])
    auc_tst = auc(fpr_tst, tpr_tst)

    return auc_tr, auc_val, auc_tst, elapsed_time


def train_logistic(XtrRDD, XvalRDD, XtstRDD):
    time_ini = time()

    # comprobando el tipo de etiquetas del dataset, deben ser 0, 1, no -1 , 1
    labels = set(XtrRDD.map(lambda x: x.label).take(100))
    #print labels
    if -1 in labels:
        XtrRDD = XtrRDD.map(lambda x: LabeledPoint((x.label + 1) / 2, x.features))
        XvalRDD = XvalRDD.map(lambda x: LabeledPoint((x.label + 1) / 2, x.features))
        XtstRDD = XtstRDD.map(lambda x: LabeledPoint((x.label + 1) / 2, x.features))
    #labels = set(XtrRDD.map(lambda x: x.label).take(100))
    #print labels
    model = LogisticRegressionWithLBFGS.train(XtrRDD)

    y_pred_trRDD = XtrRDD.map(lambda x: (x.label, model.predict(x.features)))
    y_pred_valRDD = XvalRDD.map(lambda x: (x.label, model.predict(x.features)))
    y_pred_tstRDD = XtstRDD.map(lambda x: (x.label, model.predict(x.features)))
    elapsed_time = time() - time_ini

    #labels_and_preds = test_data.map(lambda p: (p.label, logit_model.predict(p.features)))

    Ytr = y_pred_trRDD.collect()
    Yval = y_pred_valRDD.collect()
    Ytst = y_pred_tstRDD.collect()

    fpr_tr, tpr_tr, th_tr = roc_curve(np.array(Ytr)[:,0], np.array(Ytr)[:,1])
    auc_tr = auc(fpr_tr, tpr_tr)

    fpr_val, tpr_val, th_val = roc_curve(np.array(Yval)[:,0], np.array(Yval)[:,1])
    auc_val = auc(fpr_val, tpr_val)

    fpr_tst, tpr_tst, th_tst = roc_curve(np.array(Ytst)[:,0], np.array(Ytst)[:,1])
    auc_tst = auc(fpr_tst, tpr_tst)

    return auc_tr, auc_val, auc_tst, elapsed_time
