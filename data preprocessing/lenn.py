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

#df = pd.read_csv('kcore15.csv')
#df = df.head(n = 20)

#print(len(df))

df = pd.read_csv('f1.csv')
print(len(df))

df = pd.read_csv('f2.csv')
print(len(df))

df = pd.read_csv('f3.csv')
print(len(df))

df = pd.read_csv('f4.csv')
print(len(df))

df = pd.read_csv('f5.csv')
print(len(df))
 




