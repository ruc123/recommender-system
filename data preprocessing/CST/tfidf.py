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
import itertools, nltk
from nltk.stem.snowball import SnowballStemmer
#from pymongo import MongoClient
from nltk.corpus import stopwords
from string import ascii_lowercase
import re
from nltk.stem.porter import PorterStemmer

df = pd.read_csv('itemsummary.csv')


porter_stemmer = PorterStemmer()

def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    punct = str.punctuation
    words = [item for item in words if item not in punct]
    words = [porter_stemmer.stem(word) for word in words]
    words = [token for token, pos in nltk.pos_tag(words) if pos.startswith('JJ') or pos.startswith('NN')]
    return words

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word')

vect = TfidfVectorizer(stop_words='english', tokenizer=stemming_tokenizer, use_idf=False, norm='l1')
X = vect.fit_transform(df.pop('summary')).toarray()

r = df[['asin']].copy()

#del df1

df1 = pd.DataFrame(X, columns = vect.get_feature_names())

#del X
#del vect

dff = r.join(df1)

dff.to_csv('wordvec.csv', index = False)

tt = dff.columns
n = tt[1]

x = dff.sort_values('asin')

m = x.loc[:, n:]

m = m.values

m = m.transpose()

#import numpy as np

from scipy.sparse.linalg import svds

#matrix = np.random.random((10, 5))

num_components = 30

u, s, v = svds(m, k=num_components)
#X = u.dot(np.diag(s))

j = pd.DataFrame(v)

j.to_csv('V0.csv', index = False)

