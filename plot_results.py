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

kdatasets = [1, 2, 3, 4]
datasets_names = ['Ripley', 'Kwok', 'Twonorm', 'Waveform']
colors = ['b', 'r', 'g', 'k']
NCs = [5, 10, 25, 50, 100, 200]
modelo = 'hybrid'
Niter = 150

plt.clf()

for kdataset in kdatasets:
    filename = results_path + 'results_dataset_' + str(kdataset) + '_' + modelo + '.pkl'
    name = datasets_names[kdataset-1]
    with open(filename, 'r') as f:
        [aucs, times] = pickle.load(f)
    plt.plot(NCs, aucs, colors[kdataset-1], label= name)

    '''
    for i, NC in enumerate(NCs):
        filename = 'results_dataset_' + str(kdataset) + '_modelo_' + modelo + '_NC_' + str(NC) + '_Niter_' + str(Niter) + '.pkl'
        AUC = aucs[i]
        TIME = times[i]
        with open(filename, 'w') as f:
            pickle.dump([AUC, TIME], f)
    '''

plt.legend()
plt.show()

