import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.keyedvectors import KeyedVectors

class WordEmbeddingTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, wordvectors_path, use_maximum=False):
    self.use_maximum = use_maximum
    count = 100000
    self.wordvectors = KeyedVectors.load_word2vec_format(wordvectors_path, limit=count)

  def get_relevant_chars(self, tweet):
    num_hashtags = tweet.count('#')
    num_exclamations = tweet.count('!')
    num_interrogations = tweet.count('?')
    return [num_hashtags, num_exclamations, num_interrogations]

  def get_tweet_emmbeding(self, tweet):
    word_embeddings = []
    for t in tweet.split(' '):
        try:
          vec = self.wordvectors[t]
        except:
            # print('error with word "{}" reduced to "{}"'.format(self.vocab[t], s))
          pass
        else:
          word_embeddings.append(vec)

    if not len(word_embeddings):
      return np.zeros_like(self.wordvectors['a'])
    if self.use_maximum:
      return np.max(np.array(word_embeddings), axis=0)
    return np.mean(np.array(word_embeddings), axis=0)

  def transform(self, X, y=None):
    embeddigs = []
    for tweet in X:
      embeddigs.append(self.get_tweet_emmbeding(tweet))
    return np.array(embeddigs)
    # chars = []
    # for tweet in X:
    #   chars.append(self.get_relevant_chars(tweet))
    # return np.array(chars)

  def fit(self, X, y=None):
    return self