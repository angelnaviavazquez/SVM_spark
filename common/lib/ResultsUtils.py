#!/usr/bin/python
##  Functions to test a model
#
#  Functions to test a model

from KernelUtils import *
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

## Obtains the predictions of a model on a data.
#  @param data A data
#  @param Bases The set of basis elements.
#  @param Beta The weights of the model.
#  @param sigma The kernel parameter.
   
def _predict(data,Bases,Beta,sigma):
    pred = kncVector(Bases,np.array(data.features),sigma).transpose().dot(Beta)    
    print pred
    return pred[0,0] 


## Obtains the label predictions of a model on a data.
#  @param data A data
#  @param Bases The set of basis elements.
#  @param Beta The weights of the model.
#  @param sigma The kernel parameter.

def _labelAndPrediction(data,Bases,Beta,sigma):   
    return (data.label,_predict(data,Bases,Beta,sigma))


## Obtains the accuracy of the model on a dataset.
#  @param dataset The dataset
#  @param Bases The set of basis elements.
#  @param Beta The weights of the model.
#  @param sigma The kernel parameter.

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


## Copute the AUC score of the model on the training, validation and test sets.
#  @param dataset_tr Distributed training dataset.
#  @param dataset_val Distributed Validation dataset.
#  @param dataset_tst Distributed test dataset.
#  @param Bases The set of basis elements.
#  @param Pesos The weights of the model.
#  @param sigma The kernel parameter.

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



