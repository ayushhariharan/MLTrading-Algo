import bs4 as bs
import pickle
import requests
import streamlit as st
from algorithm.SessionState import *
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import time 

from algorithm.data_generation import *

import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import accuracy_score
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf

st.sidebar.title("Stock Market Prediction with Machine Learning")

def generate_linear_model(feature_df, epochs):
    X_train, X_test, y_train, y_test = split_data_linear(feature_df)
    
    regressor = KerasRegressor(build_fn=linear_model, batch_size=16,epochs=epochs)
    callback = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/Regressor_model.h5', monitor='mean_absolute_error',  
                verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

    results = regressor.fit(X_train,y_train,callbacks=[callback])

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

session_state = SessionState.get(fetch_data=False,predict=False, feature=False, model_gen=False, first_fetch=False)
tickers = save_tickers()
count = st.sidebar.selectbox(
            "How many Stocks to Consider?", (200, 300, 400, 500))
if st.sidebar.button("Fetch Data", key="fetch"):
    session_state.fetch_data = True

if session_state.fetch_data:
    fetch_data(count, session_state.first_fetch)
    session_state.first_fetch = True

    selected_stocks = tickers[:count].copy()
    stock = st.sidebar.selectbox(
        "Which Stock to Predict?", tuple(selected_stocks)
    )

    st.write(f'Chosen Stock for Analysis: {stock}')
    
    visualizations = st.sidebar.multiselect("Select Visualizations", ["Candlestick", "Moving Average", "Volume/Price Flux"])
    if st.sidebar.button("Generate Feature Set", key="feature"):
        session_state.feature = True

    if session_state.feature:
        features = generate_features(stock, visualizations)
        st.write("*Important Stock Features*")
        st.write(features.head())

        models = st.sidebar.multiselect("Select Models", ["Linear Regression"])
        epochs = st.sidebar.number_input('Number of Epochs')
        if st.sidebar.button("Train Model", key = "model"):
            session_state.model_gen = True

            if session_state.model_gen:
                if "Linear Regression" in models:
                    generate_linear_model(features, epochs)