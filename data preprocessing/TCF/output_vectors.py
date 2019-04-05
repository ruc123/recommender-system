
import pandas as pd






df1 = pd.read_csv('Dtest1.csv')
df2 = pd.read_csv('Dtrain2345_noval.csv')
df3 = pd.read_csv('Dval.csv')



df1 = df1[['reviewerID', 'asin', 'overall']]
df1.to_csv('test_vec.csv', index = False)

df2 = df2[['reviewerID', 'asin', 'overall']]
df2.to_csv('train_vec.csv', index = False)

df3 = df3[['reviewerID', 'asin', 'overall']]
df3.to_csv('probe_vec.csv', index = False)






