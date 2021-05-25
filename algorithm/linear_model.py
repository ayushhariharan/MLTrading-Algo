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

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import accuracy_score
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf

def generate_RNN_model(feature_df, epochs, already_trained):
    print("hello")

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

def split_data_linear(feature_df):
    y_df = feature_df[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']]
    y_df_mod = y_df.drop(['Volume', 'Adj Close'], axis=1)
    y = y_df_mod.values

    dropped_columns = y_df.columns.tolist()
    dropped_columns.append('Date')

    X_df = feature_df.drop(dropped_columns, axis = 1)
    X = X_df.values
    
    return train_test_split(X, y, test_size=0.3)