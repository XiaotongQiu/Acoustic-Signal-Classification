#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 16:02:23 2018

@author: qcat
"""

import filecmp
import os
import subprocess
import tempfile
import numpy as np
import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.decomposition import DictionaryLearning
from scipy.fftpack import dct
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def Generate_Y(X,n_features):
    X_f = []
    for i in X: 
#        X_f.append(np.fft.fft(i))
        X_f.append(dct(i))
    Y = np.empty([256,861])
    for i in range(80):
        x = np.array(X_f[i])
        n = np.floor(x.shape[0]/n_features)
        x = x[:int(n * n_features)]
        x = np.reshape(x,(256,-1))
        Y = np.hstack((Y,x))
    Y = Y[:,861:]
#    print(Y.shape)
    return Y

def cross_OMP(D1,D2,Y1,Y2):
    '''
    Input:
    D1: Dictionary of Train [256 * n_components]
    D2: Dictionary of Clapping  [256*components]
    Y1: Data matrix of SINGLE sample of Train [256*n]
    Y2: Data matrix of SINGLE sample of Train [256*n]
    Output:
    X1, X2: Concatenated feature vectors [2 * n_components * n]
    
    ** n = 861 **
    '''
    # X11
    omp11 = OrthogonalMatchingPursuit(n_nonzero_coefs=5)
    omp11.fit(D1,Y1)
    X11 = omp11.coef_
    # X12
    omp12 = OrthogonalMatchingPursuit(n_nonzero_coefs=5)
    omp12.fit(D1,Y2)
    X12 = omp12.coef_
    # X21
    omp21 = OrthogonalMatchingPursuit(n_nonzero_coefs=5)
    omp21.fit(D2,Y1)
    X21 = omp21.coef_
    # X22
    omp22 = OrthogonalMatchingPursuit(n_nonzero_coefs=5)
    omp22.fit(D2,Y2)
    X22 = omp22.coef_
    
    # concatenate
    X1 = np.hstack((X11,X12))
#    print(X1.shape)
    X1 = np.max(X1,axis = 0)
#    print(X1.shape)
    X2 = np.hstack((X21,X22))
    X2 = np.max(X2,axis = 0)
    return X1, X2
    
        
path_to_ESC = './ESC-50-master/audio'
#os.listdir(path_to_ESC)

y = [45,22] # train vs clapping
noise = np.random
X_data = []
X_data_noise = []
y_data = []
X_1 = []
X_2 = []
X_1_noise = []
X_2_noise = []
for file in os.listdir(path_to_ESC):
    if file.split('.')[1] == 'wav':
        if int(file.split('.')[0].split('-')[-1]) in y:
            rate, audio = wavfile.read(path_to_ESC +'/'+ file)
            audio_noise = audio.astype(np.float64)
            audio_noise += np.random.normal(scale=5, size=len(audio))
            X_data.append(audio)
            X_data_noise.append(audio_noise)
            y_data.append(file.split('.')[0].split('-')[-1])
            if int(file.split('.')[0].split('-')[-1]) == y[0]:
                X_1.append(audio)
                X_1_noise.append(audio_noise)
            else:
                X_2.append(audio)
                X_2_noise.append(audio_noise)
X_1 = np.vstack((X_1,X_1_noise))
X_2 = np.vstack((X_2,X_2_noise))

Y_1 = Generate_Y(X_1,256)
#Y_1 = Y_1[:,-861*20:]
print(Y_1.shape)
Y_2 = Generate_Y(X_2,256)
#Y_2 = Y_2[:,-861*20:]
print(Y_2.shape)
D1 = np.load('./Dict_train.npy')
D2 = np.load('./Dict_clapping.npy')
print(D1.shape,D2.shape)
#X1,X2 = cross_OMP(D1,D2,Y_1,Y_2)
n_sample = 80
X1 = np.empty([n_sample,1000])
X2 = np.empty([n_sample,1000])
for i in range(n_sample):
    y_1 = Y_1[:,i*861:(i+1)*861]
    y_2 = Y_2[:,i*861:(i+1)*861]
    x1,x2 = cross_OMP(D1,D2,y_1,y_2)
    X1[i,:] = x1
    X2[i,:] = x2
    
print(X1.shape, X2.shape)
y1 = np.ones(X1.shape[0])
y2 = np.zeros(X2.shape[0])
X = np.vstack((X1,X2))
y = np.append(y1,y2)
#X = np.max(X,axis = 0) # max pooling

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
s = clf.score(X_test,y_test)

XXX_train = preprocessing.scale(X_train)
XXX_test = preprocessing.scale(X_test)
clf_scale = svm.SVC()
clf_scale.fit(XXX_train,y_train)
acc_scale_train = clf_scale.score(XXX_train,y_train)
acc_scale_test = clf_scale.score(XXX_test,y_test)
print(acc_scale_train,acc_scale_test)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
knn_s = neigh.score(X_test,y_test) 

neigh1 = KNeighborsClassifier(n_neighbors=3)
neigh1.fit(XXX_train,y_train)
knn_sss = neigh.score(XXX_test,y_test) 

#Y_train = Y[:,-861*1:].T
#print(Y.shape)
#dict = DictionaryLearning(n_components = 500,max_iter = 10,transform_n_nonzero_coefs=5)
#D = dict.fit(Y_train).components_
#D = D.T
#print(D.shape)
#Y_test = Y[:,:861]
#omp = OrthogonalMatchingPursuit(n_nonzero_coefs=5)
#omp.fit(D,Y_test)
#coef = omp.coef_
##idx_r, = coef.nonzero()
##plt.stem(idx_r, coef[idx_r])
#plt.subplot(221)
#t=coef[0,:]
#yy = np.matmul(D,t)
#plt.plot(yy)
#plt.subplot(222)
#plt.plot(Y_test[:,0])
#plt.subplot(223)
#t=coef[1,:]
#yy = np.matmul(D,t)
#plt.plot(yy)
#plt.subplot(224)
#plt.plot(Y_test[:,1])





#            
#X_data_f = []
#X_1_f = []
#X_2_f = []
#for i in X_data:
##    print(i)
#    X_data_f.append(np.fft.fft(i))
#for i in X_1:
#    X_1_f.append(np.fft.fft(i))
#for i in X_2:
#    X_2_f.append(np.fft.fft(i))

#plt.plot(X_data_f[0])
#plt.show()