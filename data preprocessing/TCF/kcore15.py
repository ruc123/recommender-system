import numpy as np
import pandas as pd
import gzip
from pandas import Series, DataFrame
import pandas as pd
import json
import operator
from collections import Counter




# In[4]:


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
#df['newsummary'] = df[['reviewText', 'summary']].apply(lambda x: ''.join(x), axis=1)
#df['newsummary'] = df['newsummary'].str.lower()



print (len(df))

df = df.groupby('reviewerID')

df = df.filter(lambda x: len(x) > 14)

print (len(df))

df = df.groupby('asin')

df = df.filter(lambda x: len(x) > 14)

print (len(df))

#df = df[['reviewerID', 'asin', 'overall']]
df.to_csv('kcore15.csv', index = False)
