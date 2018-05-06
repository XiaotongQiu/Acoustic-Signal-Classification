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

def Generate_Y(X,n_features):
    X_f = []
    for i in X:
#        X_f.append(np.fft.fft(i))
        X_f.append(dct(i))
    Y = np.empty([256,861])
    for i in range(40):
        x = np.array(X_f[i])
        n = np.floor(x.shape[0]/n_features)
        x = x[:int(n * n_features)]
        x = np.reshape(x,(256,-1))
        Y = np.hstack((Y,x))
    Y = Y[:,861:]
#    print(Y.shape)
    return Y
        
path_to_ESC = './ESC-50-master/audio'
#os.listdir(path_to_ESC)

y = [45,22] # train vs clapping

X_data = []
y_data = []
X_1 = []
X_2 = []
for file in os.listdir(path_to_ESC):
    if file.split('.')[1] == 'wav':
        if int(file.split('.')[0].split('-')[-1]) in y:
            rate, audio = wavfile.read(path_to_ESC +'/'+ file)
            X_data.append(audio)
            y_data.append(file.split('.')[0].split('-')[-1])
            if int(file.split('.')[0].split('-')[-1]) == y[0]:
                X_1.append(audio)
            else:
                X_2.append(audio)
Y = Generate_Y(X_1,256)
Y_train = Y[:,-861*1:].T
print(Y.shape)
dict = DictionaryLearning(n_components = 500,max_iter = 10,transform_n_nonzero_coefs=5)
D = dict.fit(Y_train).components_
D = D.T
print(D.shape)
Y_test = Y[:,:861]
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=5)
omp.fit(D,Y_test)
coef = omp.coef_
#idx_r, = coef.nonzero()
#plt.stem(idx_r, coef[idx_r])
plt.subplot(221)
t=coef[0,:]
yy = np.matmul(D,t)
plt.plot(yy)
plt.subplot(222)
plt.plot(Y_test[:,0])
plt.subplot(223)
t=coef[1,:]
yy = np.matmul(D,t)
plt.plot(yy)
plt.subplot(224)
plt.plot(Y_test[:,1])



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