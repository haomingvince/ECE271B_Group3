'''
Main codes
written by Guoren Zhong
gzhong@eng.ucsd.edu
Mar.15, 2021
'''
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from DataLoader import *
from Predictions import *
from Similarity import *

#if __name__ == '__main__':
if True:
    
    # load data
    data_dir = r'clean2.csv'
    train_data, test_data = load_data(data_dir)
    labels = list(test_data['Score'])

    # get the needed inputs
    feature_matrix = pd.pivot_table(train_data, values='Score', index=['UserId'], columns=['ProductId'])
    feature_matrix.fillna(0, inplace=True)
    feature_matrix_T = pd.DataFrame.transpose(feature_matrix)

    usersPerItem = defaultdict(set)
    itemsPerUser = defaultdict(set)
    reviewsPerUser = defaultdict(list)
    reviewsPerItem = defaultdict(list)
    for row in train_data.itertuples():
        prod, user, score = row[1], row[2], row[3]
        usersPerItem[prod].add(user)
        itemsPerUser[user].add(prod)
        reviewsPerUser[user].append((prod, score))
        reviewsPerItem[prod].append((user, score))

    globalMean = sum(train_data['Score'])/len(train_data)
    meanRatings = {}
    for i, user in enumerate(feature_matrix.columns):
        ratings = feature_matrix[user]
        ratingsValid = ratings[ratings != 0]
        meanRatings[user] = np.mean(ratingsValid)

    # Make predictions
    alwaysPredictMean = [globalMean for _ in range(len(test_data))]
    cfPredictions = []
    for j in range(len(test_data.index)):
        i = test_data.index[j]
        cfPredictions.append(predictRating(test_data['ProductId'][i], test_data['UserId'][i], 
                                           globalMean, meanRatings, 
                                           usersPerItem, itemsPerUser, 
                                           reviewsPerItem, reviewsPerUser, 
                                           feature_matrix, feature_matrix_T, 
                                           base='item', simi='Jaccard'))

    cfPredictions = np.array(cfPredictions)
    cfPredictions[cfPredictions < 1] = 1
    cfPredictions[cfPredictions > 5] = 5
    err_baseline = MSE(alwaysPredictMean, labels)
    err_CF = MSE(cfPredictions, labels)
    print('For baseline =', err_baseline)
    print('The MSE of rating estimation is', err_CF)
