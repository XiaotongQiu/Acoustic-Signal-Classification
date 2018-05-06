#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:18:39 2018

@author: qcat
"""
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit

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
    X1 = np.hstack((X11,X12)).T
    X2 = np.hstack((X21,X22)).T
    return X1, X2



