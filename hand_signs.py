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

np.random.seed(1)

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
    
    return X_tr_org, X_ts_org, Y_tr_org, Y_ts_org, clss

X_tr_org, X_ts_org, Y_tr_org, Y_ts_org, classes = loading_datasets()

## normalize the features
X_tr = X_tr_org/255 
X_ts = X_ts_org/255

## coverted each output as a 6 neuron Flattened layer.
Y_tr = (np.eye(6)[Y_tr_org.reshape(-1)])
Y_ts = (np.eye(6)[Y_ts_org.reshape(-1)]) 

conv_layers = {}

## creating placeholders
def create_ph(N_h0, N_w0, N_c0, N_y0):
    
    X = tf.placeholder('float32', shape = (None, N_h0, N_w0, N_c0), name = 'X')
    Y = tf.placeholder('float32', shape = (None, N_y0), name = 'Y')
    
    return X, Y

## initializing the learning parameters
def initializing_params():
    
    tf.compat.v1.set_random_seed(1)
    W1 = tf.get_variable("W1", shape = (4, 4, 3, 8), initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", shape = (2, 2, 8, 16), initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    
    params = {"W1" : W1,
              "W2" : W2}
    return params

tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters = initializing_params()
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print("W1[1,1,1] = \n" + str(parameters["W1"].eval()[1,1,1]))
    print("W1.shape: " + str(parameters["W1"].shape))
    print("\n")
    print("W2[1,1,1] = \n" + str(parameters["W2"].eval()[1,1,1]))
    print("W2.shape: " + str(parameters["W2"].shape))

def frwd_prop(X, params):
    
    tf.compat.v1.set_random_seed(1)
    W1 = params['W1']
    W2 = params['W2']
    
    # CONV2D
    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')
    # Relu
    A1 = tf.nn.relu(Z1)
    #MAXPOOL
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    # CONV2D
    Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')
    # Relu
    A2 = tf.nn.relu(Z2)
    #MAXPOOL
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    #FLATTEN
    F = tf.contrib.layers.flatten(P2)
    #Fully connected layer
    Z3 = tf.contrib.layers.fully_connected(F, 6, activation_fn = None)
    
    return Z3

tf.reset_default_graph()

with tf.Session() as sess_1:
    np.random.seed(1)
    X, Y = create_ph(64, 64, 3, 6)
    params = initializing_params()
    Z3 = frwd_prop(X, params)
    init_1 = tf.global_variables_initializer()
    sess_1.run(init_1)
    b = sess_1.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
    print("Z3 = \n" + str(b))
## cost computation
def cost_computation(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
    
    return cost

## creating random minibatches
def create_randm_minibatches(X, Y, minibatch_size = 64, seed = 0):
    m = X.shape[0]
    minibatches = []
    #np.random.randn(seed)
    
    perm = list(np.random.permutation(m)) 
    X_shuff = X[perm,:,:,:]
    Y_shuff = Y[perm, :]
    
    minibatches_num = math.floor(m/minibatch_size)
    
    for m in range(0, minibatches_num):
        minibatch_X = X_shuff[m*minibatch_size:(m+1)*minibatch_size, :, :, :]
        minibatch_Y = Y_shuff[m*minibatch_size:(m+1)*minibatch_size, :]
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch)
        
    if m % minibatch_size != 0:
        minibatch_X = X_shuff[minibatches_num*minibatch_size:m, :, :, :]
        minibatch_Y = Y_shuff[minibatches_num*minibatch_size:m, :]
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch)
    
    return minibatches


## creating the final model

def model(X_tr, X_ts, Y_tr, Y_ts, learning_rate = 0.009, num_epochs = 100, minibatch_size = 64, print_cost = True):
    
    ops.reset_default_graph()
    tf.compat.v1.set_random_seed(1)
    seed = 3    
    (m, N_h0, N_w0, N_c0) = X_tr.shape
    N_y = Y_tr.shape[1]
    costs = []
    
    
    X, Y = create_ph(N_h0, N_w0, N_c0, N_y)
    
    params = initializing_params()
    
    Z3 = frwd_prop(X, params)
    
    cost = cost_computation(Z3, Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
    
        for e in range(0,num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m/minibatch_size)
            seed = seed + 1
            minibatches = create_randm_minibatches(X_tr, Y_tr, minibatch_size, seed)
            
            for i in minibatches:
               (minibatch_X, minibatch_Y) = i
               _, temp_cost = sess.run([optimizer, cost], {X: minibatch_X, Y: minibatch_Y})
               
               minibatch_cost += temp_cost/num_minibatches
           
            if print_cost == True and e % 5 == 0:
                print("cost after epoch %i = %f" %(e, minibatch_cost))
            if print_cost == True and e % 1 == 0:
                costs.append(minibatch_cost)
           
            plt.plot(np.squeeze(costs))
            plt.xlabel('iteration per tens')
            plt.ylabel('cost')
            plt.title('Learning rate = ' + str(learning_rate))
            plt.show()
            
            ## calculate the predictions
            prediction_output = tf.argmax(Z3, axis = 1)
            prediction_actual = tf.argmax(Y, axis = 1)
            correct_prediction = tf.equal(prediction_output, prediction_actual)
            
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            print(accuracy)
            tr_accuracy = accuracy.eval({X: X_tr, Y: Y_tr})
            ts_accuracy = accuracy.eval({X: X_tr, Y: Y_tr})
            
            print("training accuracy = " + str(tr_accuracy))
            print("testing accuracy = " + str(ts_accuracy))
            
            return tr_accuracy, ts_accuracy, params


_, _, params = model(X_tr, X_ts, Y_tr, Y_ts)
        
        
   
    
 
        
    