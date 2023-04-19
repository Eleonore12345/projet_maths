#LOW-RANK APPROXIMATION
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



### Averaging ###

#fig,axs=plt.subplots(3)
#axs[0].imshow(R,interpolation=None)
#axs[0].set_title("R")
#By row
Rr=R.copy()
for i in range(Rr.shape[0]) : 
    sum=0
    index=0
    for j in range(Rr.shape[1]):
        if Rr[i,j]!=0:
            index+=1
            sum+=Rr[i,j]
    if index==0: #making sure if the whole row consists of zeros, it remains zeros
        Rr[i,:]=0
    else :
        moy=sum/index
    for j in range(Rr.shape[1]):
        if Rr[i,j]==0:
            Rr[i,j]=moy
#axs[1].imshow(Rr,interpolation=None)
#axs[1].set_title("Rr")
#By column
Rc=R.copy()
for i in range(Rc.shape[1]) : 
    sum=0
    index=0
    for j in range(Rc.shape[0]):
        if Rc[j,i]!=0:
            index+=1
            sum+=Rc[j,i]
    if index==0: #making sure if the whole column consists of zeros, it remains zeros
        Rc[:,i]=0
    else :
        moy=sum/index
    for j in range(Rr.shape[0]):
        if Rc[j,i]==0:
            Rc[j,i]=moy
#axs[2].imshow(Rc,interpolation=None)
#axs[2].set_title("Rc")
#plt.show()

### Method featureMatrices ###
def featureMatrices(k,U,S,VH):
    """
    takes as parameters the svd decomposition and the wanted rank
    returns user feature matrix X and item feature matrix Y
    X consists of k coefficients (features) for each user linearly combined by srqt(sk)
    Y consists of k coefficients (features) for each movie linearly combined by sqrt(sk)
    """
    uk=U[:,:k]
    sk=S[:k]
    vhk=VH[:k,:]
    X=uk@np.diag(np.sqrt(sk))
    Y=np.diag(np.sqrt(sk))@vhk
    return X,Y

##### Working with Rc #####

### Computation of the reduced svd with Rc ###
u_Rc,s_Rc,vh_Rc=np.linalg.svd(Rc,full_matrices=False)
### Quality of the prediction - MAE with Rc ###
rangeK = list(range(1,31)) + [40,50]
MAE_Rc = []
trueRating = [r['rating'] for r in testUserItem] # extract the true ratings
for k in rangeK:
    # compute the truncated SVD
    X_Rc, Y_Rc = featureMatrices(k, u_Rc, s_Rc, vh_Rc)
    # perform the prediction
    prediction_Rc = [np.dot(X_Rc[(r['user']-1),:],Y_Rc[:,(r['movie']-1)]) for r in testUserItem]
    # compute the errors
    errorRating_Rc = np.asmatrix(prediction_Rc)-np.asmatrix(trueRating)
    # Get MAE
    MAE_Rc.append(np.mean(np.abs(errorRating_Rc))) # Mean Absolute Error
pl.plot(rangeK, MAE_Rc, '-+')
pl.xlabel('number of features k')
pl.ylabel('MAE with Rc')
pl.show()
#Comparison with naive predictions with Rc
naivePrediction_Rc=[]
for r in testUserItem :
    naivePrediction_Rc.append(Rc[r['user']-1,r['movie']-1])
errorRating_Rc = np.asmatrix(naivePrediction_Rc)-np.asmatrix(trueRating)
naiveMAE_Rc=np.mean(np.abs(errorRating_Rc))
kMin_Rc=int(np.where(MAE_Rc==min(MAE_Rc))[0]+1)
print("The MAE_Rc is minimal for k=",kMin_Rc)
print("with Rc, naive MAE : ",naiveMAE_Rc," MAE for k=min : ",MAE_Rc[kMin_Rc-1])


##### Working with Rr #####

### Computation of the reduced svd with Rr ###
u_Rr,s_Rr,vh_Rr=np.linalg.svd(Rr,full_matrices=False)
### Quality of the prediction - MAE with Rr ###
rangeK = list(range(1,31)) + [40,50]
MAE_Rr = []
# trueRating remains the same
for k in rangeK:
    # compute the truncated SVD
    X_Rr, Y_Rr = featureMatrices(k, u_Rr, s_Rr, vh_Rr)
    # perform the prediction
    prediction_Rr = [np.dot(X_Rr[(r['user']-1),:],Y_Rr[:,(r['movie']-1)]) for r in testUserItem]
    # compute the errors
    errorRating_Rr = np.asmatrix(prediction_Rr)-np.asmatrix(trueRating)
    # Get MAE
    MAE_Rr.append(np.mean(np.abs(errorRating_Rr))) # Mean Absolute Error
pl.plot(rangeK, MAE_Rr, '-+')
pl.xlabel('number of features k')
pl.ylabel('MAE with Rr')
pl.show()
#Case of naive predictions with Rr
naivePrediction_Rr=[]
for r in testUserItem :
    naivePrediction_Rr.append(Rr[r['user']-1,r['movie']-1])
errorRating_Rr = np.asmatrix(naivePrediction_Rr)-np.asmatrix(trueRating)
naiveMAE_Rr=np.mean(np.abs(errorRating_Rr))
kMin_Rr=int(np.where(MAE_Rr==min(MAE_Rr))[0]+1)
print("The MAE_Rr is minimal for k=",kMin_Rr)
print("with Rr, naive MAE : ",naiveMAE_Rr," MAE for k=min : ",MAE_Rr[kMin_Rr-1])

### Results ###
"""
The MAE_Rr is minimal for k= 15
with Rc, naive MAE :  0.8267336672210568  MAE for k=min :  0.7895117317767938
The MAE_Rr is minimal for k= 17
with Rr, naive MAE :  0.8501912740150434  MAE for k=min :  0.8049271515519592
"""
#There is a bigger difference with respect to the naive version for Rr however the MAE to be chosen is Rc because it is the smallest

### Robustness ###
#for u1: Rc, k=15, MAE = 0.7895
#for u2: Rc, k=13, MAE = 0.7768
#for u3: Rc, k=10, MAE = 0.7706
#for u4: Rc, k=17, MAE = 0.7703
#for u5: Rc, k=16, MAE = 0.7904
"""
The quality of our prediction is robust: Rc is always to be chosen. 
The value of k varies between 10 and 16, which can be seen as a lot
however we can see on the graphs that for values of ks in this range, the MAEs always stay low and do not vary much
"""