#NON LINEAR LEAST-SQUARES PROBLEM FORMULATION

##########Imports##########
import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import main

########## Computation of the derivatives and solving for Xu and Yi ##########
'''
dF(X(u,:))/dX(u,:) = -2Ydiag(W(u,:))R(u,:)* + 2Ydiag(W(u,:))Y*X(u,:)* + 2lambdaX(u,:)*
dG(Y(:,i))/dY(:,i) = -2X*diag(W(:,i))R(:,i) + 2X*diag(W(:,i))XY(:,i) + 2lambdaY(:,i)

dF(X(u,:))/dF(u,:) = 0    pour    X(u,:) = inv(Ydiag(W(u,:))Y* + lambda) * Ydiag(W(u,:))R(u,:)*
dG(Y(:,i))/dY(:,i) = 0    pour    Y(:,i) = inv(X*diag(W(:,i))X + lambda) * X*diag(W(:,i))R(:,i)
'''

########## Alternate least-squares algorithm ##########

def alsq(nbrIter,nbrFeatures,lmbd,R,test):

    X =np.ones((nu, nbrFeatures))
    Y = np.ones((nbrFeatures, ni))
    W0 = R > 0
    W=np.zeros(np.shape(R))
    for row in range(len(R)):
        for column in range(len(R[row])):
            if W0[row,column]==True:
                W[row,column]=1
            else:
                W[row,column]=0
    # print(W)
    MAE_alsq = []

    for ii in range(nbrIter):
        for u in range(nu):
            # Compute the least-square problem for Xu
            Wu=np.diag(W[u,:])
            # n=np.shape(Y@Wu@Y.T)[0]
            X[u]=((np.linalg.inv(Y@Wu@Y.T+lmbd*np.identity(nbrFeatures)))@Y@Wu@(R[u,:].T)).T
            # X[u] = np.linalg.solve(np.dot(Y, np.dot(Wu, Y.T)) + lmbd*np.eye(n), np.dot(Y, np.dot(Wu, R[u,:].T))).T
        for i in range(ni):
            # Compute the least-square problem for Yi
            Wi=np.diag(W[:,i])
            # n=np.shape((X.T)@Wi@X)[0]
            Y[:,i]=(np.linalg.inv((X.T)@Wi@X+lmbd*np.identity(nbrFeatures)))@(X.T)@Wi@(R[:,i])
            # Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(Wi, X)) + lmbd*np.eye(nFeat), np.dot(X.T, np.dot(Wi, R[:,i])))

        prediction = [np.dot(X[(r['user']-1),:],Y[:,(r['movie']-1)]) for r in test]
        # compute the errors
        trueRating = [r['rating'] for r in testUserItem]
        errorRating = np.asmatrix(prediction)-np.asmatrix(trueRating)
        MAE_alsq.append(np.mean(np.abs(errorRating)))

        print(ii, np.mean(np.abs(errorRating)))
        
    return X,Y,MAE_alsq


########## Data Loading ##########
baseUserItem,testUserItem=main.dataLoading()
########## Meta-Data Loading ##########
User,Item,Genre,Occupation=main.metaDataloading()
########## User-item matrix ##########
R=main.userItemMatrixCreation(User,Item,baseUserItem)
########## Defining nu and ni ##########
nu=len(User)
ni=len(Item)
########## Testing the alsq method and finding the best MAE ##########
print(alsq(5,2,0.05,R,testUserItem))