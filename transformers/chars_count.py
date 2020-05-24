import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.keyedvectors import KeyedVectors

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

class CharsCountTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, dataset_name, nrc_scores=None, use_NRC=True, count_emojis=True, count_elongated=True):
    self.relevant_chars = ['#', '!', '?']#, @]

    self.count_elongated = count_elongated
    self.use_NRC = use_NRC
    self.count_emojis = count_emojis

    if self.use_NRC:
      nrc_scores = nrc_scores[nrc_scores['association'] == dataset_name]
      self.dict_intensity = {r[0]:r[1] for r in nrc_scores.values}

    if self.count_emojis:
      self.relevant_emojis = ['\U0001f600', '\U0001f601', '\U0001f602', '\U0001f603', '\U0001f604', '\U0001f605', '\U0001f606', '\U0001f607', '\U0001f608', '\U0001f609',
                    '\U0001f610', '\U0001f611', '\U0001f612', '\U0001f613', '\U0001f614', '\U0001f615', '\U0001f616', '\U0001f617', '\U0001f618', '\U0001f619',
                    '\U0001f620', '\U0001f621', '\U0001f622', '\U0001f623', '\U0001f624', '\U0001f625', '\U0001f626', '\U0001f627', '\U0001f628', '\U0001f629',
                    '\U0001f630', '\U0001f631', '\U0001f632', '\U0001f633', '\U0001f634', '\U0001f635', '\U0001f636', '\U0001f637', '\U0001f638', '\U0001f639',
                    '\U0001f640', '\U0001f641', '\U0001f642', '\U0001f643', '\U0001f644', '\U0001f645', '\U0001f646', '\U0001f647', '\U0001f648', '\U0001f649',
                    '\U0001f650']
      self.emoji2index = {k:i for i,k in enumerate(self.relevant_emojis)}
    
  def get_relevant_chars(self, tweet):
    relevant_chars_counts = [tweet.count(c) for c in self.relevant_chars]
    upercases = len([c for c in tweet if c.isupper()])
    tweet_tokenized = word_tokenize(tweet)
    result = relevant_chars_counts + [upercases]

    if self.count_elongated:
      regex = re.compile(r"(.)\1{2}")
      elongated_words = len([word for word in tweet.split() if regex.search(word)])
      result += [elongated_words]

    if self.use_NRC:
      NRC = 0
      for t in tweet_tokenized:
        if t in self.dict_intensity:
          NRC +=  self.dict_intensity[t]
      NRC /= len(tweet_tokenized)
      result = [NRC] + result

    if self.count_emojis:
      # word_and_emoticon_count = len(re.findall(ru'\w+|[\U0001f600-\U0001f650]', s))  
      # longitud = len(tweet) 
      # emojis = self.text_has_emoji(tweet)
      relevant_emojis_counts = [0 for e in self.relevant_emojis]
      for e in re.finditer(r'[\U0001f600-\U0001f650]', tweet):
        if e in self.relevant_emojis:
          relevant_emojis_counts[self.emoji2index[e.group(0)]] += 1
      result += relevant_emojis_counts

    return result

  def transform(self, X, y=None):
      return np.array([self.get_relevant_chars(tweet) for tweet in X])

  def fit(self, X, y=None):
      return self

#for using lstm model
class gen_seq():
  def __init__(self,  path,vocab):
    self.vocab = vocab
    self.path = path

  def transform(self, X, y=None):
    seq = gen_sequence(X, self.vocab)
    seq = pad_sequences(seq,20)
    return np.array(seq)

  def fit(self, X, y=None):
      return self