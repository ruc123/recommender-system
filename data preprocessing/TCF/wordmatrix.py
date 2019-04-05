import matplotlib
matplotlib.use('Agg')
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
#from matplotlib.backends.backend_pdf import PdfPages
import itertools, nltk
from nltk.stem.snowball import SnowballStemmer
#from pymongo import MongoClient
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
df = getDF('/hdd/datasets/amazon/small_5core/reviews_Movies_and_TV_5.json.gz')


df['newsummary'] = df[['reviewText', 'summary']].apply(lambda x: ''.join(x), axis=1)


# In[73]:


stop_words = stopwords.words('english')
stop_words.extend(['may','also','zero','one','two','three','four','six','seven','eight','nine','across','its','cant','am','among','beside','however','yet','within','would']+list(ascii_lowercase))
stop = set(sorted(stop_words))
punct = str.punctuation
df['newsummary'] = df['newsummary'].str.lower().str.split()
df['newsummary'] = df['newsummary'].apply(lambda x: [item for item in x if item not in stop and item not in punct])
df['newsummary'] = df['newsummary'].apply(lambda x: [item for item in x if item not in stop])
df['newsummary'] = df['newsummary'].apply(lambda x: [token for token, pos in nltk.pos_tag(x) if pos.startswith('JJ') or pos.startswith('NN')])

a = df.groupby('asin')['newsummary'].apply(sum).to_frame()

a = a.rename(columns={'asin': 'asin', 'newsummary': 'words'})
#a = a.reset_index(level=['asin','words'])
a['words'] = a['words'].apply(lambda x: list(set(x)))
a.to_csv('wordmatrix.csv', index = True)