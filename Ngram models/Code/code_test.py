
# coding: utf-8

# In[4]:


import pandas as pd
df = pd.read_json('News_Category_Dataset.json', lines=True)


# # DATA CLEANING

# In[5]:


df['String'] = df[['headline', 'short_description']].apply(lambda x: ''.join(x), axis=1) # aggregated two columns of dataframe 
# into one i.e. String

df['String'] = df['String'].str.lower() # convert uppercase to lowercase
#df["String"] = df['String'].str.replace('[^\w\s]','')

df["String"] = df['String'].str.replace('[^a-z\s]', '') #remove punctuations

from nltk.corpus import stopwords
stop = stopwords.words('english')
df['String'] = df['String'].apply(lambda x: " ".join(x for x in x.split() if x not in stop)) #remove stopwords

#from textblob import TextBlob
#df['String'] = df['String'].apply(lambda x: str(TextBlob(x).correct())) #removal of incorrect spellings

#from nltk.stem import PorterStemmer
#st = PorterStemmer()
#df['String'] = df['String'].apply(lambda x: " ".join([st.stem(word) for word in x.split()])) #stemming words

from textblob import Word
df['String'] = df['String'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) #lemmatizing

df.to_csv('workfile.csv') #file saved in csv format in case if needed later


# # UNIGRAM MODEL
# #unigram.csv file saved

# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
x = CountVectorizer(ngram_range=(1,1), analyzer='word')
y = x.fit_transform(df['String'])
freq = sum(y).toarray()[0]
df1 = pd.DataFrame(freq, index = x.get_feature_names(), columns=['frequency'])
df1.reset_index(inplace = True)
df1.rename(columns={'index': 'words'}, inplace = True)
df1.to_csv('unigram.csv', index = False)


# # BIGRAM MODEL
# #Bigram.csv file saved

# In[7]:


#from sklearn.feature_extraction.text import CountVectorizer
#df = pd.read_csv('workfile.csv')
m = CountVectorizer(ngram_range=(2,2), analyzer='word')
n = m.fit_transform(df['String'])
frequ = sum(n).toarray()[0]
df2 = pd.DataFrame(frequ, index = m.get_feature_names(), columns=['frequency'])
df2.reset_index(inplace = True)
df2.rename(columns={'index': 'words'}, inplace=True)
df2.to_csv('Bigram.csv', index = False)


# # HISTOGRAMS FOR TOP 15 WORDS

# In[8]:


histdata1 = df1.sort_values('frequency', ascending=False).head(15)

histdata2 = df2.sort_values('frequency', ascending=False).head(15)


# In[9]:


import matplotlib.pyplot as plt
histdata1.plot('words', 'frequency', kind='bar')
plt.savefig('bar1.png', dpi = 400,  bbox_inches='tight' )
plt.close()


# In[10]:


histdata2.plot('words', 'frequency', kind='bar')
plt.savefig('bar2.png', dpi = 400,  bbox_inches='tight' )
plt.close()


# # TAG CLOUD FOR TOP 100 WORDS

# In[11]:


tag1 = df1.sort_values('frequency', ascending=False).head(100)

tag2 = df2.sort_values('frequency', ascending=False).head(100)


# In[12]:


count = {}
for i, j in tag1.values:
    count[i] = j

from wordcloud import WordCloud
wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequencies= count)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig('tagcloud_unigram.png', dpi = 400,  bbox_inches='tight')
plt.close()


# In[13]:


count1 = {}
for k, l in tag2.values:
    count1[k] = l

from wordcloud import WordCloud
wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequencies= count1)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig('tagcloud_bigram.png', dpi = 400,  bbox_inches='tight')
plt.close()


# # TFIDF SORTING BASED ON WEIGHTS (TOP 200) AND SAVED 200 MOST WEIGHED IN CSV FORMAT

# In[14]:


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

td_vec = TfidfVectorizer(stop_words='english', ngram_range= (1,1), norm = 'l1')
td_weigh = td_vec.fit_transform(df.String.dropna())
weights = np.asarray(td_weigh.mean(axis=0)).ravel().tolist()
tfidf_df = pd.DataFrame({'unigram': td_vec.get_feature_names(), 'tdidf_score': weights})
tfidf_df.to_csv('tfidf_unigram.csv')
tfidf_df1 = tfidf_df.sort_values(by='tdidf_score', ascending=False).head(100)
tfidf_df1.to_csv('tfidf_unigram_100.csv')  #first highest 100 saved


# In[15]:


td_vec1 = TfidfVectorizer(stop_words='english', ngram_range= (2,2), norm = 'l1')
td_weigh1 = td_vec1.fit_transform(df.String.dropna())
weights = np.asarray(td_weigh1.mean(axis=0)).ravel().tolist()
tfidf_df2 = pd.DataFrame({'bigrams': td_vec1.get_feature_names(), 'tdidf_score': weights})
tfidf_df2.to_csv('tfidf_bigram.csv')
tfidf_df3 = tfidf_df2.sort_values(by='tdidf_score', ascending=False).head(100)
tfidf_df3.to_csv('tfidf_bigram_100.csv') 

