#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 22:34:53 2023

@author: salim
"""

import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.utils.data as data
import numpy as np
from sklearn.model_selection import train_test_split
## Hyper-parameters


# List of all possible combinations of 4 or fewer filters
## data segmentation/windowing



def create_dataset(dataset, windowlen):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    # feature_scalling
    sc = StandardScaler()
    signal = sc.fit_transform(dataset)
    X = []
    for i in range(int(len(signal)/windowlen)):
        feature = signal[i*windowlen:(i+1)*windowlen]
        X.append(feature)
    return torch.tensor(np.array(X)).float()

def data_generator(root, batch_size, windowlen):
    print('Loading CHB-MIT Interical and preictal dataset...')
    preictal_data = pd.read_csv(root + '/preictal_data.csv')
    ictal_data = pd.read_csv(root + '/ictal_data.csv')
    
    class1 = create_dataset(preictal_data, windowlen=windowlen)
    y_1= torch.zeros(class1.shape[0],1)
    
    class2 = create_dataset(ictal_data, windowlen=windowlen)
    y_2 = torch.ones(class2.shape[0],1)

    datasets = torch.cat((class1, class2),0)
    labels = torch.cat((y_1, y_2), 0)
    
    X_train, X_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.25, shuffle=True, random_state=42)
    print(f" Shape of the Training data is {X_train.shape,}, and Testing data is {X_test.shape}" )

    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train),  batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(data.TensorDataset(X_test, y_test),  batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
