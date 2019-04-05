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



df = pd.read_csv('kcore10.csv')

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

df['SA'] = np.array([ analize_sentiment(tweet) for tweet in df['newsummary'] ])
#df.to_csv('SAdataframe.csv', index = False)

df1 = df[['reviewerID', 'asin', 'overall','newsummary', 'SA']]
df2 = df1.head(n=50)
df2.to_csv('sample1.csv', index = False)

df3 = df[['reviewerID', 'asin', 'SA']]
df3.to_csv('binmat.csv', index = False)

