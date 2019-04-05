import numpy as np
import pandas as pd
import gzip
from pandas import Series, DataFrame
import pandas as pd
import json
import operator
from collections import Counter
import random

df1 = pd.read_csv('f1.csv')


df2 = pd.read_csv('f3.csv')


df3 = pd.read_csv('f4.csv')


df4 = pd.read_csv('f5.csv')

Utrain2345 =  pd.concat([df1, df2, df3, df4], ignore_index=True)
Utrain2345.to_csv('utrain2345.csv', index = False)


df5 = pd.read_csv('f1.csv')
Dtest1 = df5.sample(frac=0.7)

rest30 = df5.drop(Dtest1.index)


Dtest1 = Dtest1.sample(frac=1)
Dtest1.to_csv('Dtest1.csv', index = False)


Dtrain_val = pd.concat([Utrain2345, rest30], ignore_index=True)
Dtrain_val = Dtrain_val.sample(frac=1)
Dtrain_val.to_csv('Dtrain2345_val.csv', index = False)

Dval = Dtrain_val.sample(frac=0.2)
Dtrain_noval 
Dval.to_csv('Dval.csv', index = False)

Dtrain_noval = Dtrain_val.drop(Dval.index)

Dtrain_noval = Dtrain_noval[['reviewerID', 'asin', 'SA']]


Dtrain_noval.to_csv('Dtrain2345_noval.csv', index = False)

Daux10 = Dtrain_noval.sample(frac=0.1)
Daux10.to_csv('Daux10.csv', index = False)

Daux20 = Dtrain_noval.sample(frac=0.2)
Daux20.to_csv('Daux20.csv', index = False)

Daux50 = Dtrain_noval.sample(frac=0.5)
Daux50.to_csv('Daux50.csv', index = False)

Daux80 = Dtrain_noval.sample(frac=0.8)
Daux80.to_csv('Daux80.csv', index = False)

Daux90 = Dtrain_noval.sample(frac=0.9)
Daux90.to_csv('Daux90.csv', index = False)

Daux100 = Dtrain_noval.sample(frac=1)
Daux100.to_csv('Daux100.csv', index = False)






