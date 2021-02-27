import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Prepration
X_Train_df = pd.read_csv('Logistic_X_Train.csv')
Y_Train_df = pd.read_csv('Logistic_Y_Train.csv')
X_Test_df = pd.read_csv('Logistic_X_Test.csv')
X_Train = X_Train_df.values
Y_Train = Y_Train_df.values
X_Test = X_Test_df.values
m_Test = X_Test.shape[0]
ones = np.ones((m_Test,1))
X_Test_ = np.hstack((X_Test,ones))

# Normalize Data
X_Train_ = (X_Train-X_Train.mean())/X_Train.std()

# Logistic Regression Implementation
def sigmoid(z):
    return 1/(1+np.exp(-1*z))

def hypothesis(X,theta):
    Y = np.dot(X,theta)
    m = Y.shape[0]
    for i in range(m):
        Y[i][0] = sigmoid(Y[i])
    return Y

def get_predicted_data(Y):
    m = Y.shape[0]
    Y_ = np.zeros((m,))
    for i in range(m):
        if Y[i][0]>0.5:
            Y_[i] = 1
        else:
            Y_[i] = 0
    return Y_

def gradient(X,Y,theta):
    m = X.shape[0]
    Y_h = hypothesis(X,theta)
    Y_ = get_predicted_data(Y_h)
    Y_ = Y_.reshape((m,1))
    grad = np.dot(X.T,Y-Y_)
    return grad/m

def likelihood(X,Y,theta):
    m = Y.shape[0]
    Y_h = hypothesis(X,theta)
    ones = np.ones((m,1))
    Y_left = Y_h.reshape((m,))
    Y_log_left = np.log2(Y_left)
    Y_log_left = Y_log_left.reshape((m,1))
    Y_right = ones-Y_h
    Y_right = Y_right.reshape((m,))
    Y_log_right = np.log2(Y_right)
    Y_log_right = Y_log_right.reshape((m,1))
    left = np.dot(Y.T,Y_log_left)
    right = np.dot((ones-Y).T,Y_log_right)
    return np.sum(left+right)/m

def error(lik):
    return -1*lik

def gradient_descent(X,Y,max_steps=100,learning_rate=0.1):
    m,n = X.shape
    ones = np.ones((m,1))
    X_ = np.hstack((X,ones))
    theta = np.zeros((n+1,1))
    error_list = []
    likelihood_list = []
    for i in range(max_steps):
        lik = likelihood(X_,Y,theta)
        err = error(lik)
        error_list.append(err)
        likelihood_list.append(lik)
        grad = gradient(X_,Y,theta)
        theta = theta + learning_rate*grad
    return theta,error_list,likelihood_list

# Making Predictions
theta,err_list,lik_list = gradient_descent(X_Train_,Y_Train)
Y_Test_h = hypothesis(X_Test_,theta)
Y_Test = get_predicted_data(Y_Test_h)

# Making csv file for our predicted data
Y_Test_df = pd.DataFrame(data=Y_Test,columns=['label'])
Y_Test_df.to_csv('chemical_classiffier.csv',index=False)






