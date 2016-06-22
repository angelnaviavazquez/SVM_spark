# -*- coding: utf-8 -*-
#!/usr/local/bin/python
"""
Created on 14/06/2016

@author: angel.navia@uc3m.es

"""

#from pyspark.mllib.regression import LabeledPoint
import numpy as np


def text2labeled(text):
    aux = text.split(',')
    label = float(aux[0])
    features = aux[1:]
    features = [float(f) for f in features]
    features = np.array(features)
    x = LabeledPoint(label, features)
    return x

with open('./data/HIGGS.csv', "r") as f:
    data = f.readlines()
'''
data_x = []
for d in data:
    aux = d.split(',')
    label = float(aux[0])
    features = aux[1:]
    features = [float(f) for f in features]
    features = np.array(features)
    x = (label, features)
    data_x.append(x)
'''
data_train = data[0:10000000]
data_val = data[10000000:10500000]
data_tst = data[10500000:11000000]

with open('/export/g2pi/SPARK/data/higgs_tr.txt', 'w') as f:
    f.writelines("%s" % l for l in data_train)

with open('/export/g2pi/SPARK/data/higgs_val.txt', 'w') as f:
    f.writelines("%s" % l for l in data_val)

with open('/export/g2pi/SPARK/data/higgs_tst.txt', 'w') as f:
    f.writelines("%s" % l for l in data_tst)

import code
code.interact(local=locals())

