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



df1 = pd.read_csv('com_movies.csv')
df2 = pd.read_csv('com_book.csv')

df1 = df1.sort_values('asin')
df1.asin = pd.factorize(df1.asin)[0] + 1

df1 = df1.sort_values('reviewerID')
df1.reviewerID = pd.factorize(df1.reviewerID)[0] + 1
df1.to_csv('com_movies1.csv', index = False)

df2 = df2.sort_values('asin')
df2.asin = pd.factorize(df2.asin)[0] + 1

df2 = df2.sort_values('reviewerID')
df2.reviewerID = pd.factorize(df2.reviewerID)[0] + 1
df2.to_csv('com_book1.csv', index = False)






#df1['newsummary'] = df1['newsummary'].str.lower().str.split()
#df2['newsummary'] = df2['newsummary'].str.lower().str.split()

#df1['newsummary'] = df1['newsummary'].apply(lambda x: [token for token, pos in nltk.pos_tag(x) if pos.startswith('JJ') or pos.startswith('NN')])

#df2['newsummary'] = df2['newsummary'].apply(lambda x: [token for token, pos in nltk.pos_tag(x) if pos.startswith('JJ') or pos.startswith('NN')])

