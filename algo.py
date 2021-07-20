import streamlit as st
import datetime as dt
import os
import pandas as pd
import time 

from algorithm.SessionState import *
from algorithm.data_generation import *
from algorithm.linear_model import *

st.sidebar.title("Stock Market Prediction with Machine Learning")


session_state = SessionState.get(fetch_data=False,predict=False, feature=False, 
    model_gen=False, first_fetch=False, y_pred_lin = [], y_test_lin = [], layers = [])

ss2 = SessionState.get(layers = [], num_layers = 0)

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

        models = st.sidebar.selectbox("Select Models", ["Linear Regression", "RNN", "Custom Model"])
        st.sidebar.subheader("Model Architecture")

        timesteps = 0

        if "Custom Model" in models:
            ss2.layers = []
            model_name = st.sidebar.text_input("Model Name")
            is_rnn = st.sidebar.checkbox("RNN Model?")

            layer_types = ["Dense", "Dropout", "BatchNormalization"]

            if is_rnn:
                timesteps = st.sidebar.number_input("Number of Timesteps")
                layer_types.append("LSTM")

            ss2.num_layers = st.sidebar.number_input("Number of Layers")

            st.sidebar.subheader("Model Layers")

            if len(ss2.layers) < ss2.num_layers:
                layer_type = st.sidebar.selectbox("Layer Type?", layer_types)

                if layer_type == "Dense":
                    activation = st.sidebar.selectbox("Activation Function", ['relu', 'tanh', 'linear'])
                    units = st.sidebar.number_input("Number of Units")
                    layer = Dense(units, activation=activation, kernel_initializer='normal')
                if layer_type == "Dropout":
                    percentage = st.sidebar.number_input("Dropout Rate")
                    layer = Dropout(percentage)
                if layer_type == "BatchNormalization":
                    layer = BatchNormalization()
                if layer_type == "LSTM":
                    units = st.sidebar.number_input("Number of Units")
                    layer = LSTM(units = units)

                if st.sidebar.button("Add Layer"):
                    ss2.layers.append(layer)
            else:
                epochs = st.sidebar.number_input('Number of Epochs')
        else:
            already_trained = st.sidebar.checkbox("Already Trained?")
            epochs = st.sidebar.number_input('Number of Epochs')
     
        st.write(ss2.layers)

        if st.sidebar.button("Train Model", key = "model"):
            session_state.model_gen = True

            if session_state.model_gen:
                if "Linear Regression" in models:
                    y_pred, y_test = generate_linear_model(features, int(epochs), already_trained, session_state.y_pred_lin, session_state.y_test_lin)
                    session_state.y_pred_lin = y_pred
                    session_state.y_test_lin = y_test
                if "RNN" in models:
                    y_test, y_pred = generate_RNN_model(features, int(epochs), already_trained, [], [])
                if "Custom Model" in models:
                    y_test, y_pred = generate_custom_model(features, ss2.layers, is_rnn, int(epochs), timesteps, model_name)