#!/usr/bin/python
# Filename: ResultsUtils.py

from KernelUtils import *
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

   
def _predict(data,Bases,Beta,sigma):
    pred = kncVector(Bases,np.array(data.features),sigma).transpose().dot(Beta)    
    print pred
    return pred[0,0] 

def _hybridpredict(data,Bases,Beta,sigma):
    totalfeatures = np.concatenate((np.array(data.features).reshape((1,len(data.features))),kncVector(Bases,np.array(data.features),sigma).transpose()),axis=1)
    pred = totalfeatures.dot(Beta)    
    print pred
    return pred[0,0] 


def _labelAndPrediction(data,Bases,Beta,sigma):   
    return (data.label,_predict(data,Bases,Beta,sigma))


def _labelAndHybridPrediction(data,Bases,Beta,sigma): 
    return (data.label,_hybridpredict(data,Bases,Beta,sigma))


def show_results(dataset,Bases,Pesos,sigma):

    LabelAndPredictions=np.array(dataset.map(lambda x:_labelAndPrediction(x,Bases,Pesos,sigma)).collect())

    fpr_tr, tpr_tr, th_tr = roc_curve(LabelAndPredictions[:,0], LabelAndPredictions[:,1])

    print "AUC",auc(fpr_tr, tpr_tr)
    plt.plot(fpr_tr, tpr_tr,'r')

    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('Resultado')
    plt.grid(True)
    plt.show()

    Success=LabelAndPredictions[:,0]*np.sign(LabelAndPredictions[:,1])

    print "Accuracy",np.sum(Success>0)/np.float(len(Success))


def compute_AUCs(dataset_tr, dataset_val, dataset_tst, Bases,Pesos,sigma):

    LabelAndPredictions=np.array(dataset_tr.map(lambda x:_labelAndPrediction(x,Bases,Pesos,sigma)).collect())
    fpr_tr, tpr_tr, th_tr = roc_curve(LabelAndPredictions[:,0], LabelAndPredictions[:,1])
    auc_tr = auc(fpr_tr, tpr_tr)

    LabelAndPredictions=np.array(dataset_val.map(lambda x:_labelAndPrediction(x,Bases,Pesos,sigma)).collect())
    fpr_val, tpr_val, th_val = roc_curve(LabelAndPredictions[:,0], LabelAndPredictions[:,1])
    auc_val = auc(fpr_val, tpr_val)

    LabelAndPredictions=np.array(dataset_tst.map(lambda x:_labelAndPrediction(x,Bases,Pesos,sigma)).collect())
    fpr_tst, tpr_tst, th_tst = roc_curve(LabelAndPredictions[:,0], LabelAndPredictions[:,1])
    auc_tst = auc(fpr_tst, tpr_tst)

    return auc_tr, auc_val, auc_tst


def compute_hybrid_AUCs(dataset_tr, dataset_val, dataset_tst, Bases,Pesos,sigma):
    
    LabelAndPredictions=np.array(dataset_tr.map(lambda x:_labelAndHybridPrediction(x,Bases,Pesos,sigma)).collect())
    fpr_tr, tpr_tr, th_tr = roc_curve(LabelAndPredictions[:,0], LabelAndPredictions[:,1])
    auc_tr = auc(fpr_tr, tpr_tr)

    LabelAndPredictions=np.array(dataset_val.map(lambda x:_labelAndHybridPrediction(x,Bases,Pesos,sigma)).collect())
    fpr_val, tpr_val, th_val = roc_curve(LabelAndPredictions[:,0], LabelAndPredictions[:,1])
    auc_val = auc(fpr_val, tpr_val)

    LabelAndPredictions=np.array(dataset_tst.map(lambda x:_labelAndHybridPrediction(x,Bases,Pesos,sigma)).collect())
    fpr_tst, tpr_tst, th_tst = roc_curve(LabelAndPredictions[:,0], LabelAndPredictions[:,1])
    auc_tst = auc(fpr_tst, tpr_tst)

    return auc_tr, auc_val, auc_tst
