import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Extracting The Data
Train_df = pd.read_csv('Train/Train.csv')
X_Test_df = pd.read_csv('Test/Test.csv')
X_Test = X_Test_df.values
Train = Train_df.values
X_Train = Train[:,:5]
Y_Train = Train[:,5]

#Normalize The Data
X_Train = (X_Train-X_Train.mean())/X_Train.std()

'''#Visualize The Data
f1 = X_Train[:,0]
f2 = X_Train[:,1]
f3 = X_Train[:,2]
f4 = X_Train[:,3]
f5 = X_Train[:,4]
plt.scatter(f5,Y_Train)
plt.show()'''

#Machine Learning Section
def hypothesis(theta,x):
    x = np.mat(x)
    return np.dot(x,theta)

def getW(X,query_point,tau):
    M = X.shape[0]
    W = np.eye(M)
    qx = np.mat(query_point)
    for i in range(M):
        xi = np.mat(X[i])
        num = (xi-qx)*(xi-qx).T
        den = -2*tau*tau
        W[i,i] = np.exp(num/den)
    return W

def closed_form_LOWESS(X_Train,Y_Train,X_Test,tau=0.1):
    M_Train = X_Train.shape[0]
    M_Test = X_Test.shape[0]
    ones_Train = np.ones((M_Train,1))
    ones_Test = np.ones((M_Test,1))
    X_Train = np.hstack((X_Train,ones_Train))
    X_Test = np.hstack((X_Test,ones_Test))
    Y_Test = np.zeros((M_Test,))
    Y_Train_ = np.zeros((M_Train,))
    X_Train = np.mat(X_Train)
    Y_Train = Y_Train.reshape((M_Train,1))
    Y_Train = np.mat(Y_Train)
    for i in range(M_Test):
        W = getW(X_Train,X_Test[i],tau)
        W = np.mat(W)
        theta = np.linalg.pinv(X_Train.T*(W*X_Train))*(X_Train.T*(W*Y_Train))
        if i==0:
            print(theta.shape)
            print(W.shape)
        Y_Test[i] = hypothesis(theta,X_Test[i])
    for i in range(M_Train):
        W = getW(X_Train,X_Train[i],tau)
        W = np.mat(W)
        theta = np.linalg.pinv(X_Train.T*(W*X_Train))*(X_Train.T*(W*Y_Train))
        Y_Train_[i] = hypothesis(theta,X_Train[i])
    return Y_Train_,Y_Test

#Calculating r2_score of our algorithm
def R2_score(Y,Y_):
    Y_avg = Y/Y.mean()
    Y = np.mat(Y)
    Y_ = np.mat(Y_)
    Y_avg = np.mat(Y_avg)
    num = (Y_-Y)*(Y_-Y).T
    den = (Y_avg-Y)*(Y_avg-Y).T
    r2_score = 1-(num/den)
    return r2_score[0][0]*100

#Cheching accuracy of our algorithm
Y_Train_,Y_Test = closed_form_LOWESS(X_Train,Y_Train,X_Test)
r2_score = R2_score(Y_Train,Y_Train_)
print(r2_score)

#Making csv file for our predicted test data
Y_Test_df = pd.DataFrame(data=Y_Test,columns=['target'])
Y_Test_df.to_csv('air_pollution_prediction.csv')





