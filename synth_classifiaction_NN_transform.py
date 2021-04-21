#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
different transformation functions
in a neural network

synthetic classifiaction problem
possibility to restrain information

python 3.7.7
numpy 1.19.2
scikit-learn 0.24.1
tensorflow 2.0.0
keras 2.3.1
matplitlib 3.3.2

author: adrienne bohlmann
"""

import numpy as np


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn import metrics, svm

from tensorflow import keras, random
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers

import matplotlib.pyplot as plt


##############################################################################
# reproducible or random?

# RANDOMNESS for checking robustness

# nR = None

# REPRODUCIBLE

# there are 3 random sources
# 1. in the creation of the data
# 2. in train-test-split
# 3. random weight initialization in model.fit
# for reproducable results fix everything
# for robustness-testing introduce randomness iteratively

# for synthetic data creation
nRdata = 22
# nRdata = None

# for train-test-split
nRtts = 22
# nRtts = None

# fix the model.fit (random weights)
random.set_seed(22)

##############################################################################
# restrain the available information?
# min: 2
# max: number of features (default = 5) = full information

n_X = 3

##############################################################################

class synth_data:
    def __init__(self
                 , samples = 300    # sample size, < 800 to observe behaviour before normal distribution from large numbers kicks in
                 , features = 5     # true number of explanatory variables for synthetic binary classifiaction problem
                 , shift = 2.0      # shift away from E(X) = 0
                 , exp = False      # if True, X = exp(X)
                 ):
        # make synthetic data
        # random n-class classification problem
        self.X, self.y = make_classification(n_samples = samples
                                             , n_features = features
                                             , shift = shift
                                             , random_state=(nRdata)
                                             )

        # exponential transform for simulating simple nonlinear relationship
        if exp == True:
            self.X = np.exp(self.X)

    # train test split
    def tts(self
           # , n_X = 2       # number of features to keep for analyses
            ):
        # train test split keeping only n_X explanatory variables
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X[:,0:n_X], self.y
                                                                                , random_state=(nRtts)
                                                                                , stratify=self.y
                                                                                )

        scaler_S = StandardScaler()
        self.scaler_S = scaler_S
        self.scaler_S.fit(self.X_train)
        self.X_train_S_scaled = scaler_S.transform(self.X_train)
        self.X_test_S_scaled = scaler_S.transform(self.X_test)

        scaler_N = Normalizer()
        self.scaler_N = scaler_N
        self.scaler_N.fit(self.X_train)
        self.X_train_N_scaled = scaler_N.transform(self.X_train)
        self.X_test_N_scaled = scaler_N.transform(self.X_test)

        scaler_QN = QuantileTransformer(output_distribution='normal', n_quantiles = 100)
        self.scaler_QN = scaler_QN
        self.scaler_QN.fit(self.X_train)
        self.X_train_QN_scaled = scaler_QN.transform(self.X_train)
        self.X_test_QN_scaled = scaler_QN.transform(self.X_test)


    def y_train(self):
        return self.y_train

    def y_test(self):
        return self.y_test

    def get_unscaled_data(self):
        return self.y_train, self.y_test, self.X_train, self.X_test

    def X_train(self):
        return self.X_train

    def X_test(self):
        return self.X_test

    def X_train_S_scaled(self):
        return self.X_train_S_scaled

    def X_test_S_scaled(self):
        return self.X_test_S_scaled

    def X_train_N_scaled(self):
        return self.X_train_N_scaled

    def X_test_N_scaled(self):
        return self.X_test_N_scaled

    def X_train_QN_scaled(self):
        return self.X_train_QN_scaled

    def X_test_QN_scaled(self):
        return self.X_test_QN_scaled

    def plot_hist(self):
        # unscaled data
        plt.hist(self.X_train[:, 0], alpha=0.6)
        plt.hist(self.X_train[:, 1], alpha=0.6)
        plt.title('unscaled training data')
        plt.show()

        # scaled data
        plt.hist(self.X_train_S_scaled[:, 0], alpha=0.6)
        plt.hist(self.X_train_S_scaled[:, 1], alpha=0.6)
        hist_title = str(self.scaler_S) + ' training data'
        plt.title(hist_title)
        plt.show()

        plt.hist(self.X_train_N_scaled[:, 0], alpha=0.6)
        plt.hist(self.X_train_N_scaled[:, 1], alpha=0.6)
        hist_title = str(self.scaler_N) + ' training data'
        plt.title(hist_title)
        plt.show()

        plt.hist(self.X_train_QN_scaled[:, 0], alpha=0.6)
        plt.hist(self.X_train_QN_scaled[:, 1], alpha=0.6)
        hist_title = str(self.scaler_QN) + ' training data'
        plt.title(hist_title)
        plt.show()


class nn_model:
    def __init__(self
                 #, n_X = 5 # dimension of input shape, MUST be == Feature-Dimension
                 , h1 = 21
                 , h2 = 5
                 , lr = 1e-3
                 ):
        # build a keras model with API

        # learning rate
        self.learning_rate = lr

        #optimizer
        opt = keras.optimizers.Adam(learning_rate = self.learning_rate)

        # model itself
        inputs = keras.Input(shape=(n_X,), name = 'input')

        hidden = layers.Dense(h1, activation='relu'
                              , kernel_initializer='he_normal'
                              , name = 'hidden1_relu')(inputs)
        hidden = layers.Dense(h2, activation='relu'
                              , kernel_initializer='he_normal'
                              , name = 'hidden2_relu')(hidden)
        #hidden = layers.Dense(7, activation='relu', name = 'hidden3_relu')(hidden)
        out = layers.Dense(2, activation='softmax', name = 'output_softmax')(hidden)

        # put it together
        self.model = keras.Model(inputs, outputs=[out], name='nn_model')
        plot_model(self.model, to_file='nn_model.png', show_shapes=True)

        # compile
        self.model.compile(optimizer=opt        # 'adam'
                          , loss=['sparse_categorical_crossentropy']
                          , metrics=['accuracy']
                          )

        print('NN created and compiled, shape saved in wd as nn_model.png')

    def fit(self
            , yy_train, yy_test, XX_train, XX_test
            , n_epochs= 30
            ):
        self.history = self.model.fit(XX_train, yy_train, epochs=n_epochs
                                      , validation_data=(XX_test, yy_test)
                                      )

    # plot loss
    def plot_loss(self):
        plt.plot(self.history.history['loss'], color='brown')
        plt.plot(self.history.history['val_loss'], color='orange')
        plt.title('model loss, train = blue, test = red')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'])
        self.plt_loss = plt
        #return self.plt_loss

    # plot accuracy
    def plot_acc(self):
        plt.plot(self.history.history['accuracy'], color='brown')
        plt.plot(self.history.history['val_accuracy'], color='orange')
        plt.title('model accuracy, train = blue, test = red')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'])
        #plt.show()
        self.plt_acc = plt
        #return self.plt_acc

##############################################################################
# implementation

# create synthetic data
data = synth_data()

# have a look
data.tts()
data.plot_hist()

# implement the keras model
model = nn_model(h1 = 33, h2 = 11)

# epochs
eps = 21

# cross validation loop
for i in range(7):
    # train test split and scale
    data.tts()

    # unscaled data
    model.fit(yy_train = data.y_train, yy_test = data.y_test
              , XX_train = data.X_train, XX_test = data.X_test
              , n_epochs = eps)

    plt.figure(num=1)
    model.plot_loss()
    plt.title('unscaled loss')

    plt.figure(num=2)
    model.plot_acc()
    plt.title('unscaled acc')

    # StandardScaler
    model.fit(yy_train = data.y_train, yy_test = data.y_test
              , XX_train = data.X_train_S_scaled, XX_test = data.X_test_S_scaled
              , n_epochs = eps)

    plt.figure(num=3)
    model.plot_loss()
    plt.title('Standard scaled loss')

    plt.figure(num=4)
    model.plot_acc()
    plt.title('Standard scaled acc')

    # Normalizer
    model.fit(yy_train = data.y_train, yy_test = data.y_test
              , XX_train = data.X_train_N_scaled, XX_test = data.X_test_N_scaled
              , n_epochs = eps)

    plt.figure(num=5)
    model.plot_loss()
    plt.title('Normalizer scaled loss')

    plt.figure(num=6)
    model.plot_acc()
    plt.title('Normalizer scaled acc')

    # Quantile Transformer (normalization)
    model.fit(yy_train = data.y_train, yy_test = data.y_test
              , XX_train = data.X_train_QN_scaled, XX_test = data.X_test_QN_scaled
              , n_epochs = eps)

    plt.figure(num=7)
    model.plot_loss()
    plt.title('QuantileTrans (norm) scaled loss')

    plt.figure(num=8)
    model.plot_acc()
    plt.title('QuantileTrans (norm) scaled acc')
