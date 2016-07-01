# -*- coding: utf-8 -*-
#!/usr/local/bin/python
"""
Created on 14/06/2016

@author: angel.navia@uc3m.es

"""

#from pyspark.mllib.regression import LabeledPoint
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
import csv


def scale_nonzero(X):
    l_X = list(X)
    N = len(l_X)
    l_X = [float(x) + 0.0000000001 if x is not None else None for x in l_X]
    #l_X = [float(x) + 0.0000000001 for x in l_X if x is not None]
    X_ = np.copy(np.reshape(l_X, (N, 1)))
    PosNonZero = np.nonzero(X_)[0]
    X_nz = X_[PosNonZero]
    x_nz = list(X_nz)
    x_nz = [float(x) for x in x_nz]
    X_nz = np.array(x_nz)
    scaler = preprocessing.StandardScaler().fit(X_nz)
    X_nz_esc = scaler.transform(X_nz)
    NL = X_nz_esc.shape[0]
    X_nz_esc_reshaped = np.copy(np.reshape(X_nz_esc, (NL, 1)))
    X_esc = np.copy(X)
    X_esc[PosNonZero] = X_nz_esc_reshaped
    return X_esc


def text2labeled(text):
    aux = text.split(',')
    label = float(aux[0])
    features = aux[1:]
    features = [float(f) for f in features]
    features = np.array(features)
    x = LabeledPoint(label, features)
    return x

df_tr = pd.read_csv('/export/g2pi/SPARK/data/kddcup1999_train', index_col=False, header=0);
df_tst = pd.read_csv('/export/g2pi/SPARK/data/kddcup1999_test', index_col=False, header=0);
#df.head(2) # devuelve n filas

x_tr = df_tr.values
X_tr = np.array(x_tr)
NPtr = X_tr.shape[0]
NI = X_tr.shape[1]

'''
# numeric
x = X_tr[:, 4]
scaler = preprocessing.StandardScaler().fit(x)
x_esc = scaler.transform(x)

# onehot
x = X_tr[:, 1]
label_binarizer = LabelBinarizer()
model = label_binarizer.fit(x)
x_ = model.transform(x)
'''

x_tst = df_tst.values
X_tst = np.array(x_tst)
NPtst = X_tst.shape[0]

list_onehot = [1, 2, 3]
list_numeric = [0, 4, 5, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
list_binary = [6, 11, 13, 20, 21]
list_labels = [41]

labels_tr = X_tr[:, 41]
cuales = (labels_tr == 'normal.')
y_tr = np.zeros((NPtr, 1))
y_tr[cuales] = 1.0
X_tr_new = y_tr

labels_tst = X_tst[:, 41]
cuales = (labels_tst == 'normal.')
y_tst = np.zeros((NPtst, 1))
y_tst[cuales] = 1.0
X_tst_new = y_tst

for kcol in range(0, 41):  # excluding labels

    if kcol in list_onehot:
        # Onehot
        x_tr = X_tr[:, kcol].reshape((NPtr, 1))
        x_tst = X_tst[:, kcol].reshape((NPtst, 1))
        label_binarizer = LabelBinarizer()
        model = label_binarizer.fit(x_tr)
        x_tr_bin = model.transform(x_tr)
        x_tst_bin = model.transform(x_tst)
        X_tr_new = np.hstack((X_tr_new, x_tr_bin))
        X_tst_new = np.hstack((X_tst_new, x_tst_bin))

    if kcol in list_binary:
        print kcol
        # Binary
        x_tr = X_tr[:, kcol].reshape((NPtr, 1))
        #print X_tr_new.shape
        #print x_tr.shape
        X_tr_new = np.hstack((X_tr_new, x_tr))
        #print X_tr_new.shape
        x_tst = X_tst[:, kcol].reshape((NPtst, 1))
        X_tst_new = np.hstack((X_tst_new, x_tst))

    if kcol in list_numeric:
        # Numeric
        x_tr = X_tr[:, kcol].reshape((NPtr, 1))
        x_tst = X_tst[:, kcol].reshape((NPtst, 1))
        scaler = preprocessing.StandardScaler().fit(x_tr)
        x_tr_esc = scaler.transform(x_tr)
        x_tst_esc = scaler.transform(x_tst)
        X_tr_new = np.hstack((X_tr_new, x_tr_esc))
        X_tst_new = np.hstack((X_tst_new, x_tst_esc))

    print X_tr_new.shape

X_tr = X_tr_new[0:4500000,:]
X_val = X_tr_new[4500000:,:]
X_tst = X_tst_new


np.savetxt('/export/g2pi/SPARK/data/kddcup1999_tr', X_tr, delimiter=",")
np.savetxt('/export/g2pi/SPARK/data/kddcup1999_val', X_val, delimiter=",")
np.savetxt('/export/g2pi/SPARK/data/kddcup1999_tst', X_tst, delimiter=",")


import code
code.interact(local=locals())


# duration: continuous.
# 1 protocol_type: symbolic.
# 2 service: symbolic.
# 3 flag: symbolic.
# src_bytes: continuous.
# dst_bytes: continuous.
# 6 land: symbolic.
# wrong_fragment: continuous.
# urgent: continuous.
# hot: continuous.
# num_failed_logins: continuous.
# 11 logged_in: symbolic.
# num_compromised: continuous.
# root_shell: continuous.
#  su_attempted: continuous.
#  num_root: continuous.
#  num_file_creations: continuous.
#  num_shells: continuous.
#  num_access_files: continuous.
#  num_outbound_cmds: continuous.
# 20 is_host_login: symbolic.
# 21 is_guest_login: symbolic.
#  count: continuous.
#  srv_count: continuous.
#  serror_rate: continuous.
#  srv_serror_rate: continuous.
#  rerror_rate: continuous.
#  srv_rerror_rate: continuous.
#  same_srv_rate: continuous.
#  diff_srv_rate: continuous.
#  srv_diff_host_rate: continuous.
#  dst_host_count: continuous.
#  dst_host_srv_count: continuous.
#  dst_host_same_srv_rate: continuous.
#  dst_host_diff_srv_rate: continuous.
#  dst_host_same_src_port_rate: continuous.
#  dst_host_srv_diff_host_rate: continuous.
#  dst_host_serror_rate: continuous.
#  dst_host_srv_serror_rate: continuous.
#  dst_host_rerror_rate: continuous.
#  dst_host_srv_rerror_rate: continuous.





with open('/export/g2pi/SPARK/data/kddcup1999_test', "r") as f:
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

