#import matplotlib
#matplotlib.use('Agg')
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
#import itertools, nltk
#from nltk.stem.snowball import SnowballStemmer
#from pymongo import MongoClient
from nltk.corpus import stopwords
from string import ascii_lowercase

df = pd.read_csv('com_movies1.csv')

#stop_words = stopwords.words('english')
#stop_words.extend(['may','also','zero','one','two','three','four','six','seven','eight','nine','across','its','cant','am','among','beside','however','yet','within','would']+list(ascii_lowercase))
#stop = set(sorted(stop_words))
#punct = str.punctuation
#df['newsummary'] = df['newsummary'].str.lower().str.split()
#df['newsummary'] = df['newsummary'].apply(lambda x: [item for item in x if item not in stop and item not in punct])
#df['newsummary'] = df['newsummary'].apply(lambda x: [item for item in x if item not in stop])
#df['newsummary'] = df['newsummary'].apply(lambda x: [token for token, pos in nltk.pos_tag(x) if pos.startswith('JJ') or pos.startswith('NN')])

df = df.groupby(['asin'])['summary'].apply(lambda x: ' '.join(x)).reset_index()
df = df.sort_values('asin')


#a = a.rename(columns={'asin': 'asin', 'newsummary': 'words'})
#a = a.reset_index(level=['asin','words'])
#a['words'] = a['words'].apply(lambda x: list(set(x)))
df.to_csv('itemsummary.csv', index = False)
print(len(df))
