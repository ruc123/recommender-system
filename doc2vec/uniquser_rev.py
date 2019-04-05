import pandas as pd

df = pd.read_csv('shuf15core.csv')

df1 = df.sample(frac=1)

df2 = df1.groupby(['reviewerID'], sort = False)['newsummary'].agg(lambda col: ' '.join(col)).reset_index()

df2.to_csv('user&rev.csv', index = False)