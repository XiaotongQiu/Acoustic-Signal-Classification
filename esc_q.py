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

def Generate_Y(X,n_features,n_sound):
    X_f = []
    for i in X: 
        X_f.append(dct(i))
    Y = np.empty([256,0])
    for i in range(n_sound):
        x = np.array(X_f[i])
        n = np.floor(x.shape[0]/n_features)
        x = x[:int(n * n_features)]
        x = np.reshape(x,(256,-1))
        Y = np.hstack((Y,x))
    return Y
def cross_OMP(D,Y, n_class,n_comp,n_sample):
    '''
    Input:
    D: List of ictionary of Train [n_class, 256 * n_components]
    Y: List of difference classes of data matrix of SINGLE sample of Train; [n_class, 256*n_samples]
    Output:
    X_set: List of concatenated feature vectors [n_class, n_components * n]
    
    ** n = 861 **
    '''
    X_set = np.zeros((n_class,n_comp*n_class))
    i = 0
    for y in Y:
        Xi = np.empty([n_sample,0])
        for d in D:
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=5)
            omp.fit(d,y)
#            print(Xi.shape,omp.coef_.shape)
            o=omp.coef_
            Xi = np.hstack((Xi,omp.coef_))
        Xi = np.max(Xi,axis = 0)
        X_set[i,:] = Xi
        i += 1
    return X_set
    
path_to_ESC = './ESC-50-master/audio'
#os.listdir(path_to_ESC)

y = [45,22,20,8,11] # train vs clapping
#y = [45,22,20,8]
n_class = len(y)

X = [[] for i in range(n_class)]

sc = 5

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
            audio_noise += np.random.normal(scale=sc, size=len(audio))
            
            for ite in range(len(y)):
                if int(file.split('.')[0].split('-')[-1]) == y[ite]:
                    X[ite].append(audio_noise)
            
#            X_data.append(audio)
#            X_data_noise.append(audio_noise)
#            y_data.append(file.split('.')[0].split('-')[-1])
#            if int(file.split('.')[0].split('-')[-1]) == y[0]:
#                X_1.append(audio)
#                X_1_noise.append(audio_noise)
#            else:
#                X_2.append(audio)
#                X_2_noise.append(audio_noise)
#X_1 = np.vstack((X_1,X_1_noise))
#X_2 = np.vstack((X_2,X_2_noise))


Y = [[] for i in range(n_class)]
D = [[] for i in range(n_class)]

n_sound = 40


for ite in range(n_class):
    Y[ite] = Generate_Y(X[ite],256,n_sound)
    print(Y[ite].shape)
    if os.path.exists('./Dict_'+str(y[ite])+'.npy'):
        D[ite] = np.load('./Dict_'+str(y[ite])+'.npy')
    else:
        print('no dict.')
        raise ValueError('no dict.')
    print('Dict. shape ', D[ite].shape)
#Y_1 = Generate_Y(X_1,256)
##Y_1 = Y_1[:,-861*20:]
#print(Y_1.shape)
#Y_2 = Generate_Y(X_2,256)
##Y_2 = Y_2[:,-861*20:]
#print(Y_2.shape)

#y = [45,22,20,8,11]
    

    
#D1 = np.load('./Dict_45.npy')
#D2 = np.load('./Dict_22.npy')
#D3 = np.load('./Dict_20.npy')
#D4 = np.load('./Dict_8.npy')
#D5 = np.load('./Dict_11.npy')
#print(D1.shape,D2.shape)
#X1,X2 = cross_OMP(D1,D2,Y_1,Y_2)
    
    

D_set = D
#n_class = len(D_set)
n_comp = 500
n_sample = 861

X_features = [np.empty([n_sound,n_comp * n_class]) for i in range(n_class)]


#X1 = np.empty([n_sound,1000])
#X2 = np.empty([n_sound,1000])


for i in range(n_sound):
    Y_set = []
    
    for Y_class in Y:
        y_single = Y_class[:,i*861:(i+1)*861]
#        y_1 = Y_1[:,i*861:(i+1)*861]
#        y_2 = Y_2[:,i*861:(i+1)*861]
    #    x1,x2 = cross_OMP(D1,D2,y_1,y_2)
        Y_set.append(y_single)
    X_set = cross_OMP(D_set,Y_set,n_class,n_comp,n_sample)
    for ite in range(n_class):
        X_features[ite][i,:] =  X_set[ite]
        
#    X1[i,:] = X_set[0]
#    X2[i,:] = X_set[1]
#    
for ite in range(n_class):
    print(str(ite), X_features[ite].shape)
    
y_label = np.empty([0])
X_data = np.empty([0,n_comp * n_class])

for ite in range(n_class):
    y_label = np.append(y_label,np.ones(X_features[ite].shape[0])*ite)
    X_data = np.vstack((X_data,X_features[ite]))
    
print('X_data: ', X_data.shape)
print('y_label: ', y_label.shape)

#y1 = np.ones(X1.shape[0])
#y2 = np.zeros(X2.shape[0])
#X = np.vstack((X1,X2))
#y = np.append(y1,y2)
#X = np.max(X,axis = 0) # max pooling

X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size = .25)
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

#neigh1 = KNeighborsClassifier(n_neighbors=3)
#neigh1.fit(XXX_train,y_train)
#knn_sss = neigh.score(XXX_test,y_test) 

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