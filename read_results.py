# -*- coding: utf-8 -*-
#!/usr/local/bin/python
"""
Created on 14/06/2016

@author: angel.navia@uc3m.es

"""

import pickle
import matplotlib.pyplot as plt
import csv
import numpy as np

# Trabajamos en local:
results_path = "./results/"

modelos = ['hybridgrad', 'kernelgrad', 'SGMA_IRWLS', 'LinearSVM', 'Logistic', 'random_IRWLS', 'hybrid_IRWLS', 'Kmeans_IRWLS']
dataset_names = ['Ripley', 'Kwok', 'Twonorm', 'Waveform', 'Adult', 'Susy', 'KddCup1999']

kdataset = 5
kfold = 0
modelo = 'SGMA_IRWLS'
modelo = 'hybrid_IRWLS'
Niter = 150
NC = 20
Samplefraction = 0.05

filename = results_path + 'dataset_' + str(kdataset) + '_modelo_' + modelo + '_NC_' + str(NC) + '_Niter_' + str(Niter) + '_kfold_' + str(kfold) + '.pkl'
with open(filename, 'r') as f:
    [auc_tr, auc_val, auc_tst, exe_time] = pickle.load(f)

print "Dataset = %s, modelo = %s, kfold = %d, Niter = %d, NC = %d" % (dataset_names[kdataset], modelo, kfold, Niter, NC)
print "AUCtr = %f, AUCval = %f, AUCtst = %f" % (auc_tr, auc_val, auc_tst)
print "Elapsed minutes = %f" % (exe_time / 60.0)
