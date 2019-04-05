import numpy as np
import pandas as pd
import gzip
from pandas import Series, DataFrame
import pandas as pd
import json
import operator
from collections import Counter
import random


df1 = pd.read_csv('com_movies1.csv')
df1 = df1.sample(frac=1)
#df1.to_csv('shuf15core.csv', index = False)

#user = df1["reviewerID"].unique()
#random.shuffle(user, random.random)

c_a = df1['asin'].nunique()
c_r = df1['reviewerID'].nunique()

print ("no. of items:   " , c_a)
print ("no. of users:   " , c_r)


#df1 = df.sample(frac=1)
#df1.to_csv('shuf15core.csv', index = False)

user = df1["reviewerID"].unique()
random.shuffle(user, random.random)

n = int(c_r/5)
n = n + 1

fiveuser = [user[i:i + n] for i in range(0, len(user), n)]

a, b, c, d, e = fiveuser



s1 = df1[df1['reviewerID'].isin(a)]
s1 = s1.sample(frac=1)
s1 = s1[['reviewerID', 'asin', 'overall']]
print(len (s1))
s1.to_csv('f1.csv', index = False)


s2 = df1[df1['reviewerID'].isin(b)]
s2 = s2.sample(frac=1)
s2 = s2[['reviewerID', 'asin', 'overall']]
print(len (s2))
s2.to_csv('f2.csv', index = False)


s3 = df1[df1['reviewerID'].isin(c)]
s3 = s3.sample(frac=1)
s3 = s3[['reviewerID', 'asin', 'overall']]
print(len (s3))
s3.to_csv('f3.csv', index = False)


s4 = df1[df1['reviewerID'].isin(d)]
s4 = s4.sample(frac=1)
s4 = s4[['reviewerID', 'asin', 'overall']]
print(len (s4))
s4.to_csv('f4.csv', index = False)


s5 = df1[df1['reviewerID'].isin(e)]
s5 = s5.sample(frac=1)
s5 = s5[['reviewerID', 'asin', 'overall']]
print(len (s5))
s5.to_csv('f5.csv', index = False)

