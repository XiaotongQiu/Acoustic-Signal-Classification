#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 00:24:24 2018

@author: qcat
"""
import os
import numpy as np
from scipy.io import loadmat
import glob
from sklearn.cross_validation import train_test_split
import os
#from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import svm
from sklearn import preprocessing

mfclist = [k for k in os.listdir('./') if k.split('.')[-1] == 'mat']

label = [45,22,20,8,11]

d = np.empty(shape=[0,39])
y = np.empty(shape=[0,1])

for i in mfclist:
    if int(i.split('-')[-1].split('.')[0]) in label:
        
        X = np.array(loadmat(i)['X'])
        d = np.concatenate([d,X])
        
        yy = np.ones_like(X[:,0]) * int(i.split('-')[-1].split('.')[0])
        y = np.append(y,yy)
sc = 0.1
d += np.random.normal(scale = sc, size = d.shape)      
print (y.shape, d.shape)
X_train, X_test, y_train, y_test = train_test_split(d, y, test_size = .25)

clf = RandomForestClassifier()
clf.fit(X_train,y_train)
s = clf.score(X_test,y_test)
print('Random Forrest Accuracy:',s)

XXX_train = preprocessing.scale(X_train)
XXX_test = preprocessing.scale(X_test)
clf_scale = svm.SVC()
clf_scale.fit(XXX_train,y_train)
acc_scale_train = clf_scale.score(XXX_train,y_train)
acc_scale_test = clf_scale.score(XXX_test,y_test)
#print(acc_scale_train,acc_scale_test)
print('SVM Accuracy:',acc_scale_test)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
knn_s = neigh.score(X_test,y_test) 
print('KNN Accuracy:',knn_s)


import matplotlib.pyplot as plt

#
#fig = plt.figure()
#plt.plot(np.array(svm_acc),'x-')
#plt.plot(np.array(rf_acc),'x-')
#plt.plot(np.array(knn_acc),'x-')
#plt.legend(['SVM','Random Forest','KNN'],loc = 'upper left')
#plt.title('Test Accuracy')
#plt.xlabel('Number of Class')
#plt.ylabel('Accuracy')
#ax = fig.add_subplot(111)
#for i in range(4):
#    
#    ax.annotate(str(rf_acc[i]), xy=(i,rf_acc[i]), xytext=(i,rf_acc[i]))
#    ax.annotate(str(svm_acc[i]), xy=(i,svm_acc[i]), xytext=(i,svm_acc[i]))
#    ax.annotate(str(knn_acc[i]), xy=(i,knn_acc[i]), xytext=(i,knn_acc[i]))