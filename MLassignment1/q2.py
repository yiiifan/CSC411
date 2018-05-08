# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)), x), axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B**2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions = np.array([LRLS(x_test[i,:].reshape(1,d),x_train,y_train, tau)\
                        for i in range(N_test)])
        #losses[j] = ((predictions-y_test)**2).mean()
        losses[j] = ((predictions.flatten() - y_test.flatten()) ** 2).mean()
    return losses
 
 
#to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a 1 x d test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    dist = l2(test_datum, x_train)
    temp = (-tau**2)*2
    B = np.min(dist)/temp
    a = np.zeros_like(dist)
    numerator = np.zeros_like(dist)
    denominator = np.sum(np.exp(dist / temp - B))

    for i in range(len(x_train)):
         numerator[0, i] = np.exp(dist[0, i] / temp - B)
         a[0, i] = numerator[0, i]/denominator
         print(i)

    A = np.diag(a[0])
    XT = x_train.T
    temp0 = np.dot(np.dot(XT, A), x_train)
    LEFT = temp0 + lam * np.eye(len(temp0))
    RIGHT = np.dot(np.dot(XT, A), y_train)

    w = np.linalg.solve(LEFT, RIGHT)
    y_hat = np.dot(test_datum, w)

    return y_hat


def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## TODO
    set_num = int(len(x) / k)

    for i in range(1,k):
        start_num = (i-1) * set_num
        end_num = i * set_num
        testset = idx[start_num : end_num]
        x_test = x[testset]
        y_test = y[testset]

        if i == 1:
            trainingset = idx[end_num+1: len(idx)]
        elif i == k:
            trainingset = idx[0:start_num - 1]
        else:
            M1 = idx[0: start_num - 1]
            N1 = idx[end_num+1: len(idx)]
            trainingset = np.hstack((M1, N1))

        x_train = x[trainingset]
        y_train = y[trainingset]
        losses = run_on_fold(x_test, y_test, x_train, y_train, taus)
        print(losses.shape)

    return losses


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish

    taus = np.logspace(1.0, 3, 200)
    print(taus)
    losses = run_k_fold(x, y, taus, k=5)
    plt.plot(losses)
    plt.xlabel("taus in [10,1000]")
    plt.ylabel("loss values")
    plt.title("Average loss against taus")
    plt.show()
    print("min loss = {}".format(losses.min()))

