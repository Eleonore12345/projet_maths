#NON LINEAR LEAST-SQUARES PROBLEM FORMULATION

##########Imports##########
import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import main

########## Data Loading ##########
baseUserItem,testUserItem=main.dataLoading()
########## Meta-Data Loading ##########
User,Item,Genre,Occupation=main.metaDataloading()
########## User-item matrix ##########
R=main.userItemMatrixCreation(User,Item,baseUserItem)
########## Defining nu et ni ##########
nu=len(User)
ni=len(Item)


########## Alternate least-squares algorithm ##########

def alsq(nbrIter,nbrFeatures,lmbd,R,test):

    X = np.ones((nu, nbrFeatures))
    Y = np.ones((nbrFeatures, ni))
    W = R > 0
    MAE_alsq = []

    for ii in range(nbrIter):
        for u in range(nu):
            # Compute the least-square problem for Xu
            Xu=(np.linalg.inv(2*Y@np.diag(W[u,:])@Y.T+2*lmbd)@2*Y@np.diag(W[u,:])@R[u,:].T).T
            np.linalg.lstsq(Xu,Y)
        for i in range(ni):
            # Compute the least-square problem for Yi
            Yi=np.linalg.inv(2*X.T@np.diag(W[:,i])@X+2*lmbd)@2*R[:,i].T@np.diag(W[:,i])@X
            np.linalg.lstsq(X,Yi)
        prediction = [np.dot(X[(r['user']-1),:],Y[:,(r['movie']-1)])
        for r in test]
        # compute the errors
        trueRating = [r['rating'] for r in testUserItem]
        errorRating = np.asmatrix(prediction)-np.asmatrix(trueRating)
        MAE_alsq.append(np.mean(np.abs(errorRating)))

        print(ii, np.mean(np.abs(errorRating)))
        
    return X,Y, MAE_alsq