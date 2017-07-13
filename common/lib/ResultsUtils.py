#!/usr/bin/python
# Filename: ResultsUtils.py

from KernelUtils import *
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import time
   
def _predict(data,Bases,Beta,sigma):
    vector = kernelMatrix(np.reshape(np.array(data.features),(1,-1)),Bases,sigma)
    pred = vector.dot(Beta)    
    return pred[0,0] 


def _labelAndPrediction(data,Bases,Beta,sigma):   
    return (data.label,_predict(data,Bases,Beta,sigma))


def compute_AUCs(dataset_tr, dataset_tst, Bases,Pesos,sigma):

    Bases=np.array(Bases)

    LabelAndPredictions=np.array(dataset_tr.map(lambda x:_labelAndPrediction(x,Bases,Pesos,sigma)).collect())
    fpr_tr, tpr_tr, th_tr = roc_curve(LabelAndPredictions[:,0], LabelAndPredictions[:,1])
    auc_tr = auc(fpr_tr, tpr_tr)
    acc_tr = compute_accuracy(LabelAndPredictions)

    t_ini = time.time()
    LabelAndPredictions=np.array(dataset_tst.map(lambda x:_labelAndPrediction(x,Bases,Pesos,sigma)).collect())
    class_time = time.time() - t_ini
    fpr_tst, tpr_tst, th_tst = roc_curve(LabelAndPredictions[:,0], LabelAndPredictions[:,1])
    auc_tst = auc(fpr_tst, tpr_tst)
    acc_tst = compute_accuracy(LabelAndPredictions)
    return auc_tr, auc_tst, acc_tr, acc_tst, class_time


def compute_accuracy(LabelAndPredictions):
    N = LabelAndPredictions.shape[0]
    aciertos = 0.0
    for k in range(0, N):
        if np.sign(LabelAndPredictions[k, 0]) == np.sign(LabelAndPredictions[k, 1]):
            aciertos += 1.0
    acc = aciertos / float(N)
    return acc
