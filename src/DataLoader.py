'''
DataLoader
written by Guoren Zhong
gzhong@eng.ucsd.edu
Mar.15, 2021
'''
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file):
    '''
    Load the data.
    :param file: directory of a csv file.
    :return: 80% train_data + 20% test_data
        
    '''
    assert os.path.splitext(file)[-1] == ".csv"

    data = pd.read_csv(file)
    data.pop('Unnamed: 0')

    train_data, test_data = train_test_split(data, test_size = 0.2, random_state=1)

    return train_data, test_data
