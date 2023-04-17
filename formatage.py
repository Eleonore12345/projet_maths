##########Imports##########
import csv
import numpy as np
import matplotlib as plt
import pylab as pl

########## Data loading ##########

baseFileName = 'ml-100k/u1.base' # training dataset
testFileName = 'ml-100k/u1.test' # test dataset
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

########## MetaData Loading ##########

userFileName = 'ml-100k/u.user'
itemFileName = 'ml-100k/u.item'
with open(userFileName, 'r') as f:
    reader = csv.DictReader(f, delimiter = '|',fieldnames=['user','age','sex','occupation','zipcode'])
    User = list(reader)
    print(len(User))
with open(itemFileName, 'r', encoding='latin1') as f:
    reader = csv.DictReader(f, delimiter = '|', fieldnames=['movie','movie title', 'release date', 'video release date','IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children''s', 'Comedy', 'Crime','Documentary', 'Drama', 'Fantasy', 'Film-Noir','Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi','Thriller', 'War', 'Western'])
    Item = [dict([key, value] for key, value in row.items()) for row in list(reader)]
    #tests d'Elolo pour essayer de comprendre des trucs
    #Itemt=list(reader)
    #print(len(Itemt))
with open('ml-100k/u.genre', 'r', encoding='latin1') as f:
    reader = csv.DictReader(f, delimiter = '|',fieldnames=['genre','genreID'])
    Genre = list(reader)
with open('ml-100k/u.occupation', 'r', encoding='latin1') as f:
    reader = csv.DictReader(f, delimiter = '|',fieldnames=['occupation'])
    Occupation = [dict([key, value] for key, value in row.items()) for row in list(reader)]