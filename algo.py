import streamlit as st
import datetime as dt
import os
import pandas as pd
import time 

from algorithm.SessionState import *
from algorithm.data_generation import *
from algorithm.linear_model import *

st.sidebar.title("Stock Market Prediction with Machine Learning")


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
        already_trained = st.sidebar.checkbox("Already Trained?")
        
        if st.sidebar.button("Train Model", key = "model"):
            session_state.model_gen = True

            if session_state.model_gen:
                if "Linear Regression" in models:
                    generate_linear_model(features, int(epochs), already_trained)