import bs4 as bs
import pickle
import requests
import streamlit as st
import SessionState
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import time 

st.sidebar.title("Stock Market Prediction with Machine Learning")
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
                df = web.DataReader(ticker, 'yahoo', start_date, end_date)
                df.to_csv('stock_details/{}.csv'.format(ticker))
            except:
                st.write("Stock {} does not exist".format(ticker))
                continue    
        else:
            continue

    
    st.write('Completed Data Collection')

session_state = SessionState.get(fetch_data=False,predict=False, first_fetch=False)
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

    if st.sidebar.button("Predict", key="predict"):
        st.write("hello")

