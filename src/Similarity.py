'''
Similarity rules
written by Guoren Zhong
gzhong@eng.ucsd.edu
Mar.15, 2021
'''
import numpy as np

def Jaccard(s1, s2):
    '''
    Compute the Jaccard similarity.
    :param s1: set 1                (set)
    :param s2: set 2                (set)
    :return: Jaccard coefficient    (float)
    '''
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom


def Cosine(x, y):
    '''
    Compute cosine similarity.
    :param x: vector 1              (set)
    :param y: vector 2              (set)
    :return: Cosine coefficient     (float)
    '''
    x[x == 1] = -1
    y[y == 1] = -1
    x[x == 2] = -1
    y[y == 2] = -1
    x[x >= 3] = 1
    y[y >= 3] = 1   
    return (x @ y.T) / (np.linalg.norm(x) * np.linalg.norm(y))


def Pearson(x, y):
    '''
    Compute Pearson correlation coefficient.
    :param x: vector 1              (set)
    :param y: vector 2              (set)
    :return: Pearson coeff.         (float)
    '''
    x_valid = x[x != 0]
    y_valid = y[y != 0]
    mean_x = np.mean(x_valid)
    mean_y = np.mean(y_valid)
    x[x != 0] -= mean_x
    y[y != 0] -= mean_y
    if (np.linalg.norm(x) * np.linalg.norm(y)) == 0: return 0
    else: return (x @ y.T) / (np.linalg.norm(x) * np.linalg.norm(y))
