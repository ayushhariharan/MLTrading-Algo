import bs4 as bs
import pickle
import requests
import streamlit as st

st.sidebar.title("Stock Market Prediction with ML")
def save_tickers():
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = bs.BeautifulSoup(resp.text)
    table = soup.find('table', {'class': 'wikitable sortable'}) #gets the table element from wikipedia

    tickers = []
    for row in table.findAll('tr')[1:]: #iterates through the row of the table
        ticker = row.findAll('td')[0].text[:-1] #Finds the symbol for that specific row
        tickers.append(ticker) #Adds it to the ticker value
    st.write(tickers)

    with open("tickers.pickle", 'wb') as f:
        pickle.dump(tickers, f)
    
    return tickers

save_tickers()