#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 19:11:44 2018

@author: qcat
"""
import numpy as np
from scipy.io import loadmat
import glob
from sklearn.cross_validation import train_test_split
import os
#from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier


def generate_dataset(labels,data_path,flag='sub_band_filter_adj_15_dict_wgt'):
    """ Create the data set with data and corresponding labels
    Input:
        (List of strings) labels : all labels
        (String) data_path : path
        (String) flag : weighted or unweighted
    Output:
        np.arrary: dataset
    """
#    dataset = []
    d = np.empty(shape=[0,39])
    y = np.empty(shape=[0,1])
    m = 0
    for i in labels:
        p = data_path  + i
        num = len(glob.glob1(p,'*.mfc'))
#        d = np.empty(shape=[0,43])    # data for 1 class
        pp = p
        if not (os.path.exists(pp)):
            continue
        mat = [k for k in os.listdir(pp) if k.split('.')[-1] == 'mat']
        for j in range(num):
#            print(os.listdir(pp))
            pp = p
            pp = pp + '/' + mat[j]
#            print(pp)
            X = np.array(loadmat(pp)['X'])
#            print(pp)
            d = np.concatenate([d,X])
            yy = np.ones_like(X[:,0]) * m
            y = np.append(y,yy)
#            print(y.shape)
        #dataset = np.concatenate([dataset,d])
#        dataset.append(d)
#        print(i)
        m += 1

    return d, y

# Generate dataset
y = ['ambulance','casino','inside_vehicle','nature_daytime','nature_night','ocean','playgrounds','police-sirens','rain','restaurant']
data_path = './dataset/'
X, y = generate_dataset(y,data_path)
#print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)

mode = RandomForestClassifier()
mode.fit(X_train,y_train)
rf = mode.score(X_test,y_test)

print(rf)

# XX_train = preprocessing.normalize(X_train, norm='l2')
# XX_test = preprocessing.normalize(X_test, norm='l2')
# clf_norm = svm.SVC()
# clf_norm.fit(XX_train, y_train)
# acc_norm_train = clf_norm.score(XX_train,y_train)
# acc_norm_test = clf_norm.score(XX_test,y_test)
# print(acc_norm_train,acc_norm_test)
#
# XXX_train = preprocessing.scale(X_train)
# XXX_test = preprocessing.scale(X_test)
# clf_scale = svm.SVC()
# clf_scale.fit(XXX_train,y_train)
# acc_scale_train = clf_scale.score(XXX_train,y_train)
# acc_scale_test = clf_scale.score(XXX_test,y_test)
# print(acc_scale_train,acc_scale_test)
