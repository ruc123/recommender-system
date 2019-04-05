import numpy as np
import pandas as pd
import gzip
from pandas import Series, DataFrame
import pandas as pd
import json
import itertools
import matplotlib.pyplot as plt
import operator
import string as str
from collections import Counter
import itertools, nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from string import ascii_lowercase


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

#df1 = getDF('/hdd/datasets/amazon/small_5core/reviews_Movies_and_TV_5.json.gz')
#df2 = getDF('/hdd/datasets/amazon/small_5core/reviews_Books_5.json.gz')


df1 = pd.read_csv('kcore15_newsum.csv')
df2 = pd.read_csv('kcore15books_newsum.csv')

#df2['newsummary'] = df2[['reviewText', 'summary']].apply(lambda x: ''.join(x), axis=1)
#df2.to_csv('kcore15books_newsum.csv', index = False)





#df1['newsummary'] = df1[['reviewText', 'summary']].apply(lambda x: ''.join(x), axis=1)
#df2['newsummary'] = df2[['reviewText', 'summary']].apply(lambda x: ''.join(x), axis=1)


#df1.asin = pd.factorize(df1.asin)[0] + 1
#df1.reviewerID = pd.factorize(df1.reviewerID)[0] + 1
#df1.to_csv('num_useritem_mov.csv', index = False)

#df2.asin = pd.factorize(df1.asin)[0] + 1
#df2.reviewerID = pd.factorize(df1.reviewerID)[0] + 1
#df2.to_csv('num_useritem_book.csv', index = False)

common = set.intersection(set(df1.reviewerID), set(df2.reviewerID))

print(len(common))

df1 = df1[df1.reviewerID.isin(common)]

df2 = df2[df2.reviewerID.isin(common)]

#df1['newsummary'] = df1[['reviewText', 'summary']].apply(lambda x: ''.join(x), axis=1)

print(len(df1))

df1.to_csv('com_movies.csv', index = False)

#df2['newsummary'] = df2[['reviewText', 'summary']].apply(lambda x: ''.join(x), axis=1)

print(len(df2))

df2.to_csv('com_book.csv', index = False)






