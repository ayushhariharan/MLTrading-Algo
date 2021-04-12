from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
import re
import ssl 

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download(['punkt', 'wordnet'])

def load_data():
    data = pd.read_csv('./data/Combined_News_DJIA.csv')
    data.fillna(data.median, inplace=True)
    return data

def create_dataset(dataset):
    dataset = dataset.drop(columns=['Date', 'Label'])
    dataset.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

    for col in dataset.columns:
        dataset[col] = dataset[col].str.lower()
    
    headlines = []
    for row in range(0, len(dataset.index)):
        headlines.append(' '.join(str(x) for x in dataset.iloc[row]))
    
    df = pd.DataFrame(headlines, columns = ['headlines'])
    data = load_data()
    
    df['label'] = data.Label
    df['date'] = data.Date

    return df

df = load_data()
df = create_dataset(df)