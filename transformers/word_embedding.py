import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.keyedvectors import KeyedVectors

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

class WordEmbeddingTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, wordvectors, aggregation='None'):
    self.wordvectors = wordvectors
    self.aggregation = aggregation

  def get_embeddings(self, tweet):
    word_embeddings = []
    for t in word_tokenize(tweet):
      try:
        vec = self.wordvectors[t]
      except:
          # print('error with word "{}" reduced to "{}"'.format(self.vocab[t], s))
        pass
      else:
        word_embeddings.append(vec)
    return word_embeddings

  def agregate(self, word_embeddings):
    if not len(word_embeddings):
      return np.zeros_like(self.wordvectors['a'])
    if self.aggregation == 'max':
      return np.max(np.array(word_embeddings), axis=0)
    if self.aggregation == 'mean':
      return np.mean(np.array(word_embeddings), axis=0)
    return word_embeddings

  def transform(self, X, y=None):
    embeddigs = []
    for tweet in X:
      embeddigs.append(self.agregate(self.get_embeddings(tweet)))
    return np.array(embeddigs)
    # chars = []
    # for tweet in X:
    #   chars.append(self.get_relevant_chars(tweet))
    # return np.array(chars)

  def fit(self, X, y=None):
    return self