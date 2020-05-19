import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
# from train_embeddings import *

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils, to_categorical


class SadnessTweetClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, vocab=None):
    # self.classifier = MultinomialNB()
    self.classifier = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter = 1000)
    # self.classifier = RandomForestClassifier()
    # self.classifier = lstm_class(vocab)

  def fit(self, X, y):
    # Check that X and y have correct shape
    X, y = check_X_y(X, y, accept_sparse=True)
    # Store the classes seen during fit
    self.classes_ = unique_labels(y)

    self.X_ = X
    self.y_ = y

    self.classifier.fit(X, y)

    # Return the classifier
    return self

  def predict_proba(self, X):
    # Check is fit had been called
    check_is_fitted(self)
    # Input validation
    X = check_array(X, accept_sparse=True)

    # return np.random.rand(X.shape[0],3)
    return self.classifier.predict_proba(X)

  def predict(self, X):
    # Check is fit had been called
    check_is_fitted(self)

    # Input validation
    X = check_array(X)

    # closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
    # return self.y_[closest]
    return self.classifier.predict(X)


class SadnessTweetDeepClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, wordvectors, max_len=15, patience=10):
    self.wordvectors = wordvectors
    self.max_len = max_len

    # Create callbacks
    self.callbacks = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]

  def get_embeddings(self, X):
    X_ = []
    for tweet in X:
      word_embeddings = []
      for i, t in enumerate(word_tokenize(tweet)):
        if i == self.max_len:
          break
        try:
          vec = self.wordvectors[t]
        except:
            # print('error with word "{}" reduced to "{}"'.format(self.vocab[t], s))
          pass
        else:
          word_embeddings.append(np.expand_dims(vec, 0))
      while len(word_embeddings) < self.max_len:
        word_embeddings.append(np.expand_dims(np.zeros_like(self.wordvectors['a']), 0))
      word_embeddings = np.expand_dims(np.concatenate(word_embeddings, axis=0), 0)
      X_.append(word_embeddings)
    X_ = np.concatenate(X_, axis=0)
    return X_
  
  def fit(self, X_train, y_train, X_valid=None, y_valid=None, epochs=200, lr=0.0001, batch_size=16):
    # Check that X and y have correct shape
    # X, y = check_X_y(X, y)
    # Store the classes seen during fit
    # self.classes_ = unique_labels(y_train)

    self.X_ = X_train
    self.y_ = y_train

    self.model = Sequential()

    # Recurrent layer
    self.model.add(Bidirectional(LSTM(128, return_sequences=False, 
            dropout=0.1, recurrent_dropout=0.1)))

    # Fully connected layer
    self.model.add(Dense(128, activation='relu'))

    # Dropout for regularization
    self.model.add(Dropout(0.5))

    # Output layer
    self.model.add(Dense(3, activation='softmax'))

    # scheduler
    lr_schedule = ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True)

    # optimizer
    opt = Adam(learning_rate=lr_schedule)

    # Compile the model
    self.model.compile(
        optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', AUC()])

    # encode class values as integers
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    self.classes_ = label_encoder.classes_
    encoded_y_train = label_encoder.transform(y_train)

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y_train = np_utils.to_categorical(encoded_y_train)  

    X_train_ = self.get_embeddings(X_train)

    if X_valid is not None:
      encoded_y_valid = label_encoder.transform(y_valid)
      dummy_y_valid = np_utils.to_categorical(encoded_y_valid)  
      X_valid_ = self.get_embeddings(X_valid)

      self.history_ = self.model.fit(X_train_,  dummy_y_train, 
                      batch_size=batch_size, epochs=epochs,
                      callbacks=self.callbacks,
                      validation_data=(X_valid_, dummy_y_valid))
    else:
      self.history_ = self.model.fit(X_train_,  dummy_y_train, 
                      batch_size=batch_size, epochs=epochs,
                      callbacks=self.callbacks)
    return self

  def predict_proba(self, X):
    # Check is fit had been called
    check_is_fitted(self)
    # Input validation
    # X = check_array(X, accept_sparse=True)

    X_ = self.get_embeddings(X)

    predicted_probs = self.model.predict(X_)
    return predicted_probs
  
  def predict(self, X):
    # Check is fit had been called
    check_is_fitted(self)
    # Input validation
    X = check_array(X, accept_sparse=True)

    return self.model.eval(X)


class SadnessTweetNRCClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, nrc_scores):
    self.nrc_scores = nrc_scores

  def fit(self, X, y):
    # Check that X and y have correct shape
    # X, y = check_X_y(X, y)
    # Store the classes seen during fit
    self.classes_ = unique_labels(y)

    self.X_ = X
    self.y_ = y

    return self

  def predict_proba(self, X):
    # Check is fit had been called
    check_is_fitted(self)
    # Input validation
    # X = check_array(X, accept_sparse=True)

    probs = []
    for tweet in X:
      low_s, low_c = 0, 1e-8
      medium_s, medium_c = 0, 1e-8
      high_s, high_c = 0, 1e-8
      for i, t in enumerate(word_tokenize(tweet)):
        try:
          score = self.nrc_scores[t]
        except:
          pass
        else:
          if score < 0.3333:
            low_s += score
            low_c += 1
          elif score < 0.6666:
            medium_s += score - 0.3333
            medium_c += 1
          else:
            high_s += score - 0.6666
            high_c += 1
      sum_s = low_s + medium_s + high_s
      if sum_s:
        probs.append([low_s/sum_s, medium_s/sum_s, high_s/sum_s])
      else:
        probs.append([0.3333, 0.3333, 0.3333])
    return np.array(probs)

  def predict(self, X):
    # Check is fit had been called
    check_is_fitted(self)
    # Input validation
    # X = check_array(X, accept_sparse=True)
    
    probs = self.predict_proba(X)
    return np.argmax(probs, axis=1)