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

import yfinance as yf
yf.pdr_override()



def save_tickers():
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = bs.BeautifulSoup(resp.text)
    table = soup.find('table', {'class': 'wikitable sortable'}) #gets the table element from wikipedia

    tickers = []
    for row in table.findAll('tr')[1:]: #iterates through the row of the table
        ticker = row.findAll('td')[0].text[:-1] #Finds the symbol for that specific row
        tickers.append(ticker) #Adds it to the ticker value

    with open("tickers.pickle", 'wb') as f:
        pickle.dump(tickers, f)
    
    return tickers    

def fetch_data(num_count, not_first):
    with open("tickers.pickle", 'rb') as f:
        tickers = pickle.load(f)
    if not os.path.exists('stock_details'):
        os.makedirs('stock_details')
    
    start_date = dt.datetime(2010, 1, 1)
    yesterday = dt.date.today() - dt.timedelta(days=1)
    end_date = yesterday - dt.timedelta(days=yesterday.weekday())
    count = 0
    num_missed = 0

    st.markdown('### Task Progress')
    st.write('Fetching Data from Yahoo Finance for {} Stocks...'.format(num_count))
    if not_first:
        latest_iteration = st.empty()
        latest_iteration.text(f'Stock {tickers[num_count - 1]} | Iteration {num_count}')
        bar = st.progress(1.0)
        st.write('Completed Data Collection')
        return
        
    latest_iteration = st.empty()
    bar = st.progress(0)

    for ticker in tickers:
        if count == num_count:
            break
        count += 1
        latest_iteration.text(f'Stock {ticker} | Iteration {count}')
        bar.progress(count / num_count)

        if not os.path.exists('stock_details/{}.csv'.format(ticker)):
            try:
                df = web.get_data_yahoo(ticker, start_date, end_date)
                df.to_csv('stock_details/{}.csv'.format(ticker))
            except:
                st.write("Stock {} does not exist".format(ticker))
            
        else:
            df = pd.read_csv('stock_details/{}.csv'.format(ticker))
            last_date = df.iloc[-1]['Date']
            if not last_date == str(end_date):
                try:
                    df = web.get_data_yahoo(ticker, start_date, end_date)
                    df.to_csv('stock_details/{}.csv'.format(ticker))
                except:
                    st.write("Stock {} does not exist".format(ticker))    
    st.write('Completed Data Collection')

def generate_features(selected_stock, visualizations):
    with open("tickers.pickle", 'rb') as f:
        tickers = pickle.load(f)
    
    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        csv_path = "stock_details/{}.csv".format(ticker)
        if selected_stock in ticker:
            continue
        if not os.path.exists(csv_path):
            continue
        stock_df = pd.read_csv(csv_path)
        stock_df.set_index('Date', inplace=True)

        stock_df.rename(columns={'Adj Close': ticker}, inplace=True)
        stock_df.drop(['Open','High','Low',"Close",'Volume'],axis=1,inplace=True)

        if main_df.empty:
            main_df = stock_df
        else:
            main_df = main_df.join(stock_df, how = 'outer')
    
    df = pd.read_csv(f'stock_details/{selected_stock}.csv')
    df.set_index('Date', inplace=True)    

    if "Candlestick" in visualizations:
        st.markdown("*Candlestick Plot*")
        our_df = pd.read_csv(f'stock_details/{selected_stock}.csv', index_col=0,parse_dates=True)
        our_df_ohlc = our_df['Adj Close'].resample('10D').ohlc()
        our_df_volume = our_df['Volume'].resample('10D').sum()
        our_df_ohlc.reset_index(inplace=True)
        our_df_ohlc['Date'] = our_df_ohlc['Date'].map(mdates.date2num)

        ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1) 
        ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=5, colspan=1, sharex=ax1)  

        candlestick_ohlc(ax1, our_df_ohlc.values, width=2, colorup='g')
        ax2.fill_between(our_df_volume.index.map(mdates.date2num), our_df_volume.values, 0)

        st.pyplot(plt.gcf())
    
    df['Moving_av'] = df['Adj Close'].rolling(window=50, min_periods=0).mean()

    if "Moving Average" in visualizations:
        st.markdown("*Moving Average Plot*")
        fig, ax = plt.subplots()
        df.plot(kind='line', y = 'Moving_av', ax = ax)
        st.pyplot(fig)
    
    i = 1
    vol_rate_increase = [0]
    price_rate_increase = [0]

    while i < len(df):
        vol_rate_increase.append(df.iloc[i]['Volume'] - df.iloc[i - 1]['Volume'])
        price_rate_increase.append(df.iloc[i]['Adj Close'] - df.iloc[i - 1]['Adj Close'])
        i += 1
    
    df['Increase_in_vol'] = vol_rate_increase
    df['Increase_in_adj_close'] = price_rate_increase

    if "Volume/Price Flux" in visualizations:
        st.markdown("*Volume Flux Plot*")
        fig, ax = plt.subplots()
        df.plot(kind='line', y = 'Increase_in_vol', ax = ax)
        st.pyplot(fig)

        st.markdown('*Price Flux Plot*')
        fig2, ax2 = plt.subplots()
        our_df.plot(kind='line', y = 'Increase_in_adj_close', ax = ax2)
        st.pyplot(fig2)

    feature_df = df.join(main_df,how='outer')
    feature_df.fillna(0.0, inplace=True)
    feature_df.reset_index(inplace=True)
    
    return feature_df
