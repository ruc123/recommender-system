from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import Word
import pandas as pd

df = pd.read_csv('shuf15core.csv')
print (len(df))


def dataprocess(message, lower_case = True, stem = True, stop_words = True, lem = True):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    #if lem:
        #words = [Word(word).lemmatize() for word in words]
    if stem:
        stemmer = PorterStemmer()
        #sno = nltk.stem.SnowballStemmer('english')
        words = [stemmer.stem(word) for word in words]
    words = ' '.join(words)
    return words

df['newsummary'] = df['newsummary'].apply(dataprocess)
df.to_csv('each_rev_process.csv', index = False)

