##########Imports##########
import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

########## Data loading ##########
def dataLoading ():
    baseFileName = 'ml-100k/u5.base' # training dataset
    testFileName = 'ml-100k/u5.test' # test dataset
    # (complement of the test dataset)
    # Load training and test datasets
    with open(baseFileName, 'r') as f:
        fieldnames=['user','movie','rating','datestamp']
        reader = csv.DictReader(f, delimiter = '\t',fieldnames=fieldnames)
        # create a dict out of reader, converting all values to integers
        baseUserItem = [dict([key, int(value)] for key, value in row.items()) for row in list(reader)]

    with open(testFileName, 'r') as f:
        fieldnames=['user','movie','rating','datestamp']
        reader = csv.DictReader(f, delimiter = '\t', fieldnames=fieldnames)
        # create a dict out of reader, converting all values to integers
        testUserItem = [dict([key, int(value)] for key, value in row.items()) for row in list(reader)]
    return baseUserItem,testUserItem

########## MetaData Loading ##########

userFileName = 'ml-100k/u.user'
itemFileName = 'ml-100k/u.item'
def metaDataloading():
    with open(userFileName, 'r') as f:
        reader = csv.DictReader(f, delimiter = '|',fieldnames=['user','age','sex','occupation','zipcode'])
        User = list(reader)
    with open(itemFileName, 'r', encoding='latin1') as f:
        reader = csv.DictReader(f, delimiter = '|', fieldnames=['movie','movie title', 'release date', 'video release date','IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children''s', 'Comedy', 'Crime','Documentary', 'Drama', 'Fantasy', 'Film-Noir','Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi','Thriller', 'War', 'Western'])
        Item = [dict([key, value] for key, value in row.items()) for row in list(reader)]
    with open('ml-100k/u.genre', 'r', encoding='latin1') as f:
        reader = csv.DictReader(f, delimiter = '|',fieldnames=['genre','genreID'])
        Genre = list(reader)
    with open('ml-100k/u.occupation', 'r', encoding='latin1') as f:
        reader = csv.DictReader(f, delimiter = '|',fieldnames=['occupation'])
        Occupation = [dict([key, value] for key, value in row.items()) for row in list(reader)]
    return User,Item,Genre,Occupation

########## User-item matrix ##########

def userItemMatrixCreation(User,Item,baseUserItem):
    nu=len(User)
    ni=len(Item)
    R = np.zeros((nu,ni)) # nu is the total number of users, ni the total number of items
    for row in baseUserItem:
        R[row['user']-1,row['movie']-1] = row['rating']
    if(R.shape[0]!=943 or R.shape[1]!=1682):
        print("Check your code, there is an error of dimension in the user-item matrix")
    return R

