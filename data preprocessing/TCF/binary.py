import numpy as np
import pandas as pd
import gzip
from pandas import Series, DataFrame
import pandas as pd
import json
import operator
from collections import Counter
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer



df1 = pd.read_csv('Daux10.csv')

df2 = pd.read_csv('Daux20.csv')

df3 = pd.read_csv('Daux50.csv')

df4 = pd.read_csv('Daux80.csv')

df5 = pd.read_csv('Daux90.csv')

df6 = pd.read_csv('Daux100.csv')



def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(tweet)
    #analysis = TextBlob(clean_tweet(tweet))
    #if  ss['pos'] > ss['neg'] && ss['':
        #return 1
    #elif analysis.sentiment.polarity == 0:
        #return 0
    #if ss['neg'] >= 0.11 and ss['neu'] <= 8 and ss['pos'] <= 0.1 :
        #return 0
    #else:
        #return 1
    if  ss['pos'] > ss['neg']:
        return 1
    else:
        return 0

df1['SA'] = np.array([ analize_sentiment(tweet) for tweet in df1['newsummary'] ])
df2['SA'] = np.array([ analize_sentiment(tweet) for tweet in df2['newsummary'] ])
df3['SA'] = np.array([ analize_sentiment(tweet) for tweet in df3['newsummary'] ])
df4['SA'] = np.array([ analize_sentiment(tweet) for tweet in df4['newsummary'] ])
df5['SA'] = np.array([ analize_sentiment(tweet) for tweet in df5['newsummary'] ])
df6['SA'] = np.array([ analize_sentiment(tweet) for tweet in df6['newsummary'] ])

bin10 = df1[['reviewerID', 'asin', 'SA']]
bin10.to_csv('bin10.csv', index = False)

bin20 = df1[['reviewerID', 'asin', 'SA']]
bin20.to_csv('bin20.csv', index = False)

bin50 = df1[['reviewerID', 'asin', 'SA']]
bin50.to_csv('bin50.csv', index = False)

bin80 = df1[['reviewerID', 'asin', 'SA']]
bin80.to_csv('bin80.csv', index = False)

bin90 = df1[['reviewerID', 'asin', 'SA']]
bin90.to_csv('bin90.csv', index = False)

bin100 = df1[['reviewerID', 'asin', 'SA']]
bin100.to_csv('bin100.csv', index = False)


