'''
Make predictions
written by Guoren Zhong
gzhong@eng.ucsd.edu
Mar.15, 2021
'''
import numpy as np
from Similarity import *


def predictRating(prod, user,
                  globalMean, meanRatings,
                  usersPerItem, itemsPerUser,
                  reviewsPerItem, reviewsPerUser,
                  feature_matrix, feature_matrix_T,
                  base='item', simi='Cosine'):
    '''
    Predict the rating of a user to a product.
    :param prod:                product id (str)
    :param user:                user id (str)
    :param globalMean:          global mean rating in the training set (float)
    :param meanRatings:         mean ratings for users (dict) 
    :param usersPerItem:        sets of all users of an item (dict)
    :param itemsPerUser:        sets of all items of a user (dict)
    :param reviewsPerItem:      reviews of each item (list)
    :param reviewsPerUser:      reviews of each user (list)
    :param feature_matrix:      rows == userID, columns == productID
    :param feature_matrix_T:    rows == productID, columns == userID
    :param base:                type of collaborative filter ("item" or "user")
    :param simi:                the similarity rule applied ("Jaccard" or "Cosine" or "Pearson")
    :return:                    rating prediction
    '''

    assert base == 'item' or base == 'user'
    assert simi == 'Jaccard' or simi == 'Cosine' or simi == 'Pearson'
    
    scores = []
    similarities = []

    if base == 'item':
        for cur_prod, cur_score in reviewsPerUser[user]:
            if cur_prod == prod: continue
            scores.append(cur_score)
            
            try:
                if simi == 'Jaccard':
                    similarities.append(Jaccard(usersPerItem[prod], usersPerItem[cur_prod]))
                elif simi == 'Cosine':
                    similarities.append(Cosine(np.array(feature_matrix[prod]), np.array(feature_matrix[cur_prod])))
                else:
                    similarities.append(Pearson(np.array(feature_matrix[prod]), np.array(feature_matrix[cur_prod])))
            except:
                similarities.append(0)

        if sum(similarities) != 0:
            weightedScores = [(x*y) for x,y in zip(scores, similarities)]
            return sum(weightedScores) / np.sum(np.abs(similarities))

        else: return globalMean


    else:
        users = []
        for cur_user, cur_score in reviewsPerItem[prod]:
            if cur_user == user: continue
            scores.append(cur_score)
            
            try:
                if simi == 'Jaccard':
                    similarities.append(Jaccard(itemsPerUser[user], itemsPerUser[cur_user]))
                elif simi == 'Cosine':
                    similarities.append(Cosine(np.array(feature_matrix_T[user]), np.array(feature_matrix_T[cur_user])))
                else:
                    similarities.append(Pearson(np.array(feature_matrix_T[user]), np.array(feature_matrix_T[cur_user])))
            except:
                similarities.append(0)

            users.append(cur_user)

        if sum(similarities) != 0:
            if user not in meanRatings:
                weightedScores = [(x*y) for x,y in zip(scores, similarities)]
                rating_pred = sum(weightedScores) / np.sum(np.abs(similarities))
            else:
                weightedScores = [((x-meanRatings[u])*y) if u in meanRatings else 0 for x,y,u in zip(scores, similarities, users)]
                rating_pred = meanRatings[user] + sum(weightedScores) / np.sum(np.abs(similarities))
            return rating_pred
        
        else: return globalMean


def MSE(predictions, labels):
    '''
    Computing the MSE of rating predictions.
    :param predictions: rating predictions.
    :param labels: rating labels
    '''
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)
