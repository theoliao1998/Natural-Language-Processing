import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from collections import Counter

def tokenize(s):
    return s.split()

def sigmoid(X):
    return 1.0 / (1.0+np.exp(-X))

def log_likelihood(B,X,Y):
    return np.sum(Y*np.inner(B,X)-np.log(1+np.exp(np.inner(B,X))),axis=0)

def compute_gradient(X,Y,Y_predict):
    return X.dot(Y-Y_predict)

def logistic_regression(X,Y,learning_rate,num_step,compute_ll=False,interval=1):
    X = np.append(X,np.ones((X.shape[0],1)),1)
    B = np.zeros(X.shape[1])
    if compute_ll:
        ll = [log_likelihood(B,X,Y)]

    for step in range(num_step):
        i = randrange(Y.shape[0])
        B += learning_rate * compute_gradient(X[i],Y[i],sigmoid(np.inner(B,X[i])))
        if compute_ll and (step+1) % interval == 0:
            ll.append(log_likelihood(B,X,Y))

    return B,ll if compute_ll else B

def predict(x,B):
    return 0 if sigmoid(np.inner(B,x)) <= 0.5 else 1