import pandas as pd
import numpy as np
import nltk
import re
import gensim
 
#from gensim.models import Doc2Vec
#from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize
#from gensim.models.deprecated.doc2vec import LabeledSentence
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

df = pd.read_csv('user&rev.csv')

class TaggedDocumentIterator(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield TaggedDocument(words=doc.split(), tags=[self.labels_list[idx]])


docLabels = list(df['reviewerID'])
data = list(df['newsummary'])
sentences = TaggedDocumentIterator(data, docLabels)


model = Doc2Vec(vector_size=125, window=10, min_count=5, epochs=100, dm = 0, sampling_threshold = 1e-5, negative_size = 5)
model.build_vocab(sentences)
model.train(sentences,total_examples=model.corpus_count, epochs=model.epochs)

model.save("d2v_user.model")

model= Doc2Vec.load("d2v_user.model")

df['vector'] = df['newsummary'].apply(model.infer_vector)

