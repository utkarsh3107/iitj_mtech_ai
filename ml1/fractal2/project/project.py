import pandas as pd
import csv
import re
import spacy

df = pd.read_csv("training.1600000.processed.noemoticon.csv",quoting=csv.QUOTE_ALL,encoding='latin-1',header=None)
df.columns = ['Sentiment','ID','Date','Query','User_ID','Text']
df.head()

df['Sentiment'].plot()
df['Sentiment'].describe()
df['Sentiment'].fillna(value=df['Sentiment'].mode(),inplace=True)

df.dropna(subset=['Text'],inplace=True)
target = df.pop(item='Sentiment')

nlp = spacy.load('en_core_web_lg')

def remove_hashtags(sentence):
    return re.sub(r'[^a-zA-Z]', " ",sentence)

def remove_usernames(sentence):
    return re.sub(r'@[A-Z0-9a-z_:]+','',sentence)

def remove_retweet_tags(sentence):
    return re.sub(r'^[RT]+','',sentence)

def remove_urls(sentence):
    return re.sub(r'(http|https|ftp)?://[A-Za-z0-9./]+','',sentence)

def clean_tweet(tweet):
    tweet = remove_usernames(tweet)
    tweet = remove_retweet_tags(tweet)
    tweet = remove_urls(tweet)
    tweet = remove_hashtags(tweet)
    return tweet.strip()

def remove_stopwords(tweet):
    return " ".join([word for word in tweet.split(" ") if word.lower() not in nlp.Defaults.stop_words])

def preproccess_pipeline(tweet):
    tweet = clean_tweet(tweet)
    tweet = remove_stopwords(tweet)
    return tweet

df['Text'] = df['Text'].apply(preproccess_pipeline)
df.head()

#https://github.com/explosion/spaCy/issues/7962
#https://github.com/ethan-carlson/Sentimentalamp/blob/master/sentimentalamp_serial.py
#https://towardsdatascience.com/creating-the-twitter-sentiment-analysis-program-in-python-with-naive-bayes-classification-672e5589a7ed