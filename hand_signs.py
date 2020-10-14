# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import h5py
import math
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from tensorflow.python.framework import ops

def loading_datasets():
    tr_dataset = h5py.File('datasets/train_signs.h5',"r")
    X_tr_org = np.array(tr_dataset["train_set_x"][:])
    Y_tr_org = np.array(tr_dataset["train_set_y"][:])
    ts_dataset = h5py.File('datasets/test_signs.h5',"r")
    X_ts_org = np.array(ts_dataset["test_set_x"][:])
    Y_ts_org = np.array(ts_dataset["test_set_y"][:])
    clss = np.array(ts_dataset["list_classes"][:])
    Y_tr_org = Y_tr_org.reshape((1, Y_tr_org.shape[0]))
    Y_ts_org = Y_ts_org.reshape((1, Y_ts_org.shape[0]))
    
    return X_tr_org, X_ts_org, Y_tr_org, Y_ts_org

X_tr_org, X_ts_org, Y_tr_org, Y_ts_org = loading_datasets()

## normalize the features
X_tr = X_tr_org/255 
X_ts = X_ts_org/255

## coverted each output as a 6 neuron Flattened layer.
Y_tr = (np.eye(6)[Y_tr_org.reshape(-1)])
Y_ts = (np.eye(6)[Y_ts_org.reshape(-1)]) 

conv_layers = {}

## creating placeholders
def create_ph

