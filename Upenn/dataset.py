#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 22:34:53 2023

@author: salim
"""

import torch
import os
from scipy.io import loadmat
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
import torch.utils.data as data
import numpy as np
from sklearn.model_selection import train_test_split
## Hyper-parameters


# List of all possible combinations of 4 or fewer filters
## data segmentation/windowing



inlabels = ['Dog_1','Dog_2','Dog_3','Dog_4','Patient_1','Patient_2','Patient_3','Patient_4','Patient_5','Patient_6','Patient_7','Patient_8']

def process(subject):

    dn = './data/%s/'%inlabels[subject]
    fns = [fn for fn in os.listdir(dn) if '.mat' in fn]

    dataset = []
    labels = []

    print(dn)
    # For each participants, resample to 400Hz.
    for fn in fns:   

        if 'inter' in fn:
            m = loadmat(dn+fn)
            d = m['data']
            d = resample(d, 400, axis=1)
            
            # feature_scalling
            sc = StandardScaler()
            d = sc.fit_transform(d)
            
            labels.append(0)
            dataset.append(d.T)
        elif '_ictal' in fn:
            m = loadmat(dn+fn)
            d = m['data']
            d = resample(d, 400, axis=1)
            
            # feature_scalling
            sc = StandardScaler()
            d = sc.fit_transform(d)
            
            labels.append(1)
            dataset.append(d.T)
        else:
            pass

    dataset = np.array(dataset)
    labels = np.array(labels)
    return torch.tensor(dataset).float(), torch.tensor(labels).float()


# In[101]:
def data_generator(inlabels, batch_size):
    dataset, labels = process(inlabels) # select the participants number (0 to 11) to process the data and further use for classification
    print("Shape of the dataset is :", dataset.shape)
    labels = labels.reshape(len(labels),1)
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.25, shuffle=True, random_state=42)
    print(f" Shape of the Training data is {X_train.shape,}, and Testing data is {X_test.shape}" )

###### Data loader #######
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train),  batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(data.TensorDataset(X_test, y_test),  batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
