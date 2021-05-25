import bs4 as bs
import pickle
import requests
import streamlit as st
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import time 
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import accuracy_score
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

def generate_RNN_model(feature_df, epochs, already_trained):
    df_train, df_test = split_data_RNN(feature_df)

    sc = MinMaxScaler(feature_range = (0, 1))

    df_target = df_train[['High','Low','Open','Close']]
    target_set = df_target.values
    train_set = df_train.values

    training_set_scaled = sc.fit_transform(train_set)
    target_set_scaled = sc.fit_transform(target_set)

    X_train = []
    y_train = []
    for i in range(50,len(train_set)):
        X_train.append(training_set_scaled[i-50:i,:])
        y_train.append(target_set_scaled[i,:])
    
    X_train, y_train = np.array(X_train), np.array(y_train)

    model = rnn_model(X_train.shape[1])




def generate_linear_model(feature_df, epochs, already_trained, y_pred, y_test):
    if len(y_pred) == 0:    
        X_train, X_test, y_train, y_test = split_data_linear(feature_df)
        regressor = KerasRegressor(build_fn=linear_model, batch_size=16,epochs=epochs)
        if already_trained:
            regressor.load_weights('checkpoints/Regressor_model.h5')  
        else:
            callback = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/Regressor_model.h5', monitor='mean_absolute_error',  
                    verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

            results = regressor.fit(X_train,y_train,callbacks=[callback])
        y_pred = regressor.predict(X_test)

    st.markdown("*Linear Regression*")

    fig, ax = plt.subplots()
    plt.figure(figsize=(20,10))
    plt.plot(y_test[:32], color = 'green', label = 'Real Stock')
    plt.plot(y_pred[:32], color = 'red', label = 'Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Trading Day')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    st.pyplot(plt.gcf())
    return y_pred, y_test
    
def rnn_model(LSTM_rows):
    mod = Sequential()
    mod.add(LSTM(units=64, return_sequences=True, input_shape=(LSTM_rows, 9)))
    mod.add(Dropout(0.2))
    mod.add(BatchNormalization())
    mod.add((LSTM(units=64)))
    mod.add(Dropout(0.1))
    mod.add(BatchNormalization())
    mod.add(Dense(units=16, activation='tanh'))
    mod.add(BatchNormalization())
    mod.add(Dense(units=4, activation='tanh'))
    mod.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy','mean_squared_error'])
    mod.summary()
    return mod

def linear_model():
    mod=Sequential()
    mod.add(Dense(32, kernel_initializer='normal',input_dim = 200, activation='relu'))
    mod.add(Dense(64, kernel_initializer='normal',activation='relu'))
    mod.add(Dense(128, kernel_initializer='normal',activation='relu'))
    mod.add(Dense(256, kernel_initializer='normal',activation='relu'))
    mod.add(Dense(4, kernel_initializer='normal',activation='linear'))
    
    mod.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy','mean_absolute_error'])
    mod.summary()
    
    return mod

def split_data_RNN(feature_df):
    df_main = feature_df[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close', 'Moving_av', 'Increase_in_vol', 'Increase_in_adj_close']]
    st.write(len(df_main))

    train_split = int(0.83 * len(df_main))
    df_train = df_main[:train_split]
    df_test = df_main[train_split:]

    return df_train, df_test


def split_data_linear(feature_df):
    y_df = feature_df[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']]
    y_df_mod = y_df.drop(['Volume', 'Adj Close'], axis=1)
    y = y_df_mod.values

    dropped_columns = y_df.columns.tolist()
    dropped_columns.append('Date')

    X_df = feature_df.drop(dropped_columns, axis = 1)
    X = X_df.values
    
    return train_test_split(X, y, test_size=0.3)